use std::process::{Command, Stdio};
use eyre::eyre;
use small_talk::config::SharedConfig;

#[derive(clap::Args, Debug)]
pub struct AllTalkCommand;

impl AllTalkCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let batch_file = r"G:\TTS\alltalk_tts\start_alltalk.bat";

        // Execute the batch file
        let process = Command::new("cmd")
            .args(["/C", batch_file])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn();

        match process {
            Ok(mut child) => {
                if let Err(e) = child.wait() {
                    eprintln!("Error waiting for the batch file process to complete: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to execute batch file: {}", e);
            }
        }

        return Ok(());

        // Change directory
        let working_dir = r"G:\TTS\alltalk_tts";
        if let Err(e) = std::env::set_current_dir(working_dir) {
            eyre::bail!("Failed to change direcotry: {}", e);
        }
        // Prepare the command to run everything in a single `cmd` subprocess
        let activate_and_run = format!(
            r#"
        call "G:\TTS\alltalk_tts\alltalk_environment\conda\condabin\conda.bat" activate "G:\TTS\alltalk_tts\alltalk_environment\env" ^&^& python script.py
        "#
        );

        // Execute the command
        let process = Command::new("cmd")
            .args(["/C", &activate_and_run])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn();

        match process {
            Ok(mut child) => {
                if let Err(e) = child.wait() {
                    eyre::bail!("Error waiting for the process to complete: {}", e);
                }
            }
            Err(e) => {
                eyre::bail!("Failed to execute command: {}", e);
            }
        }

        // // Set environment variables
        // std::env::set_var("CONDA_ROOT_PREFIX", r"G:\TTS\alltalk_tts\alltalk_environment\conda");
        // std::env::set_var("INSTALL_ENV_DIR", r"G:\TTS\alltalk_tts\alltalk_environment\env");
        //
        // // Activate the conda environment
        // let conda_activate = Command::new("cmd")
        //     .args([
        //         "/C",
        //         r"G:\TTS\alltalk_tts\alltalk_environment\conda\condabin\conda.bat",
        //         "activate",
        //         r"G:\TTS\alltalk_tts\alltalk_environment\env",
        //     ])
        //     .output();
        //
        // match conda_activate {
        //     Ok(output) => {
        //         if !output.status.success() {
        //             eyre::bail!(
        //                 "Failed to activate conda environment: {}",
        //                 String::from_utf8_lossy(&output.stderr)
        //             );
        //         }
        //     }
        //     Err(e) => {
        //         eyre::bail!(
        //                 "Error running conda activation: {}", e
        //             );
        //     }
        // }
        //
        // // Run the Python script
        // let python_script = Command::new("python")
        //     .arg("script.py")
        //     .stdout(Stdio::inherit())
        //     .stderr(Stdio::inherit())
        //     .spawn();
        //
        // match python_script {
        //     Ok(mut child) => {
        //         if let Err(e) = child.wait() {
        //             eprintln!("Failed to wait for Python script: {}", e);
        //         }
        //     }
        //     Err(e) => {
        //         eprintln!("Failed to execute Python script: {}", e);
        //     }
        // }

        Ok(())
    }
}