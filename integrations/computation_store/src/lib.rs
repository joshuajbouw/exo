use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use astrid_core::dirs::AstridHome;
use astrid_storage::{
    ContentName, KvQuotaResolver, RuntimePrincipalStore, StateOwner, open_runtime_principal_store,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::runtime::{Builder, Runtime};

const KV_NAMESPACE: &str = "system:exo-computation-reuse";
const READ_SIZE: u64 = 8 * 1024 * 1024;
const HASH_BUFFER_SIZE: usize = 32 * 1024 * 1024;

#[pyclass]
struct ComputationStore {
    runtime: Runtime,
    store: RuntimePrincipalStore,
}

#[pymethods]
impl ComputationStore {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(runtime_error)?;
        let quota: Arc<dyn KvQuotaResolver<StateOwner>> = Arc::new(|_: &StateOwner| Ok(None));
        let store = runtime
            .block_on(open_runtime_principal_store(
                &AstridHome::from_path(path),
                quota,
            ))
            .map_err(runtime_error)?;
        Ok(Self { runtime, store })
    }

    fn contains(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let name = content_name(name)?;
        py.detach(|| {
            self.store
                .content()
                .describe(&StateOwner::System, &name)
                .map(|value| value.is_some())
                .map_err(runtime_error)
        })
    }

    fn put_file(&self, py: Python<'_>, name: &str, source: PathBuf) -> PyResult<(String, String)> {
        let name = content_name(name)?;
        py.detach(|| {
            let file = File::open(source).map_err(io_error)?;
            let mut reader = HashingReader::new(BufReader::with_capacity(HASH_BUFFER_SIZE, file));
            let content = self.store.content();
            let outcome = content
                .put_streaming(&StateOwner::System, &name, &mut reader)
                .map_err(runtime_error)?;
            content.flush().map_err(runtime_error)?;
            Ok((
                object_id_hex(outcome.descriptor().file().as_bytes()),
                reader.digest(),
            ))
        })
    }

    fn get_file(
        &self,
        py: Python<'_>,
        name: &str,
        destination: PathBuf,
    ) -> PyResult<Option<(String, String)>> {
        let name = content_name(name)?;
        py.detach(|| {
            let content = self.store.content();
            let Some(handle) = content
                .open_read(&StateOwner::System, &name)
                .map_err(runtime_error)?
            else {
                return Ok(None);
            };
            let object_id = object_id_hex(handle.descriptor().file().as_bytes());
            let mut output = BufWriter::new(File::create(destination).map_err(io_error)?);
            let mut hasher = blake3::Hasher::new();
            let logical_bytes = handle.descriptor().logical_bytes();
            let mut offset = 0_u64;
            while offset < logical_bytes {
                let length = READ_SIZE.min(logical_bytes - offset);
                let bytes = handle.read_range(offset, length).map_err(runtime_error)?;
                hasher.update_rayon(&bytes);
                output.write_all(&bytes).map_err(io_error)?;
                offset += length;
            }
            output.flush().map_err(io_error)?;
            Ok(Some((object_id, tagged_blake3(hasher.finalize()))))
        })
    }

    fn digest_file(&self, py: Python<'_>, source: PathBuf) -> PyResult<String> {
        py.detach(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update_mmap_rayon(source).map_err(io_error)?;
            Ok(tagged_blake3(hasher.finalize()))
        })
    }

    fn delete(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let name = content_name(name)?;
        py.detach(|| {
            self.store
                .content()
                .delete(&StateOwner::System, &name)
                .map_err(runtime_error)
        })
    }

    fn get<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let value = py.detach(|| {
            self.runtime
                .block_on(self.store.kv().get(KV_NAMESPACE, key))
                .map_err(runtime_error)
        })?;
        Ok(value.map(|bytes| PyBytes::new(py, &bytes)))
    }

    fn set(&self, py: Python<'_>, key: &str, value: &[u8]) -> PyResult<()> {
        let value = value.to_vec();
        py.detach(|| {
            self.runtime
                .block_on(self.store.kv().set(KV_NAMESPACE, key, value))
                .map_err(runtime_error)
        })
    }

    fn remove(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        py.detach(|| {
            self.runtime
                .block_on(self.store.kv().delete(KV_NAMESPACE, key))
                .map_err(runtime_error)
        })
    }
}

struct HashingReader<R> {
    inner: R,
    hasher: blake3::Hasher,
}

impl<R> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
        }
    }

    fn digest(&self) -> String {
        tagged_blake3(self.hasher.finalize())
    }
}

fn tagged_blake3(digest: blake3::Hash) -> String {
    format!("blake3:{digest}")
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        // The content builder requests relatively small slices. Spawning Rayon
        // work for each callback costs more than hashing these slices inline.
        self.hasher.update(&buffer[..read]);
        Ok(read)
    }
}

fn content_name(value: &str) -> PyResult<ContentName> {
    ContentName::new(value).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn object_id_hex(id: &[u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in id {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn io_error(error: std::io::Error) -> PyErr {
    PyIOError::new_err(error.to_string())
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn exo_computation_store(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ComputationStore>()
}
