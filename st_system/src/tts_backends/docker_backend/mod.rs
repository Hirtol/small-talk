use bollard::{
    models::{ContainerSummary, DeviceRequest, HostConfig},
    Docker,
};
use eyre::ContextCompat;
use futures::StreamExt;
use std::{collections::HashMap, time::Duration};
use bollard::query_parameters::{StartContainerOptions, StopContainerOptions};
use crate::timeout::DroppableState;

pub mod docker_utils;

pub use docker_utils::DockerTtsCreateConfig;

pub struct DockerTemporaryState {
    pub docker_container: ContainerSummary,
    pub api_address: String,
    daemon: Docker,
}

impl DroppableState for DockerTemporaryState {
    type Context = DockerTtsCreateConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        #[tracing::instrument(skip(daemon))]
        async fn start_container(daemon: &Docker, config: DockerTtsCreateConfig) -> eyre::Result<ContainerSummary> {
            tracing::debug!(image=?config.image_name, container=?config.container_name, "Attempting to start docker container");
            let container = docker_utils::find_or_create_container(daemon, config.clone()).await?;

            daemon.start_container(container.id.as_deref().unwrap(), None::<StartContainerOptions>).await?;
            // Need to query again as we might get a randomly assigned IP address
            let final_container = docker_utils::find_or_create_container(daemon, config).await?;

            Ok(final_container)
        }

        let daemon = bollard::Docker::connect_with_local_defaults()?;
        let container = start_container(&daemon, context.clone()).await?;

        let container_port = if let Some(ports) = &container.ports {
            ports.first().and_then(|p| p.public_port).unwrap_or(context.internal_port)
        } else {
            context.internal_port
        };
        let api_address = format!("http://localhost:{container_port}");
        tracing::debug!(?api_address, "Started container");

        Ok(DockerTemporaryState {
            daemon,
            api_address,
            docker_container: container,
        })
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        self.daemon.stop_container(self.docker_container.id.as_deref().unwrap(), None::<StopContainerOptions>).await?;
        Ok(())
    }
}