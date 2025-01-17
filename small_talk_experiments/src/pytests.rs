//! Conclusion: Can't really use PyO3 because it compiles _for_ a specific VENV, it can't adapt to run-time VENVs.
use std::env;
use std::ffi::CString;
use std::fs::File;

use fundsp::hacker::*;
use fundsp::wave::Wave;
use pyo3::ffi::c_str;
use pyo3::{PyResult, Python};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyModule;
use wavers::Wav;

pub fn main() -> eyre::Result<()> {
    let current_path = env::var("PATH")?;
    let new_path = format!("{};{}", r"G:\TTS\Kokoro-tts\.venv\Scripts\", current_path);
    // let new_path = format!("{};{}", r"C:\Users\Valentijn\AppData\Roaming\uv\python\cpython-3.12.7-windows-x86_64-none", current_path);

    unsafe {
        env::set_var("VIRTUAL_ENV", r"G:\TTS\Kokoro-tts\.venv\");
        env::set_var("PATH", &new_path);
    }
    let current_path = env::var("PATH")?;
    println!("{}", current_path);
    let code = std::fs::read_to_string(r"G:\TTS\Kokoro-tts\hello.py")?;

    let m = Python::with_gil(|py| -> PyResult<_> {
        let sys = PyModule::import(py, "sys").unwrap();
        let path = sys.getattr("path")?;
        path.call_method1("append", ("G:\\TTS\\Kokoro-tts\\.venv\\Lib\\site-packages",))?;  // append my venv path
        let py_path: String = sys.getattr("executable").unwrap().extract().unwrap();
        let py_version: String = sys.getattr("version").unwrap().extract().unwrap();

        let sysconfig = PyModule::import(py, "sysconfig").unwrap();
        let python_version = sysconfig.call_method0("get_python_version").unwrap();
        println!("Using python version: {}", python_version);
        let python_lib = sysconfig
            .call_method("get_config_var", ("LIBDEST",), None)
            .unwrap();
        println!("Using python lib: {}", python_lib);
        let python_site_packages = sysconfig
            .call_method("get_path", ("purelib",), None)
            .unwrap();
        println!("Using python site-packages: {}", python_site_packages);

        println!("executable: {}\nversion: {}", py_path, py_version);
        let activators = PyModule::from_code(py, CString::new(code)?.as_c_str(), c_str!("hello.py"), c_str!("hello"))?;

        let fs = activators.getattr("main")?.call0()?.unbind();
        Ok(fs)
    })?;

    Ok(())
}