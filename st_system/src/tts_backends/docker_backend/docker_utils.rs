use bollard::{
    config::ContainerCreateBody,
    models::{ContainerSummary, DeviceRequest, HostConfig},
    query_parameters::{CreateContainerOptions, CreateImageOptions, ListContainersOptions},
    Docker
    ,
};
use eyre::ContextCompat;
use futures::StreamExt;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct DockerTtsCreateConfig {
    pub container_name: String,
    pub image_name: String,
    pub internal_port: u16,
}

/// Find an existing container, or create a new instance of it based on the provided `config`.
pub async fn find_or_create_container(
    daemon: &Docker,
    config: DockerTtsCreateConfig,
) -> eyre::Result<ContainerSummary> {
    let container = find_container(daemon, &config.container_name).await?;

    if let Some(container) = container {
        Ok(container)
    } else {
        create_container(daemon, config).await
    }
}

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),* $(,)?) => {{
        let mut map = std::collections::HashMap::new();
        $( map.insert($key, $val); )*
        map
    }};
}

pub async fn create_container(daemon: &Docker, create_conf: DockerTtsCreateConfig) -> eyre::Result<ContainerSummary> {
    // First pull the image if it doesn't exist.
    let _ = daemon
        .create_image(
            Some(CreateImageOptions {
                from_image: Some(create_conf.image_name.clone()),
                ..Default::default()
            }),
            None,
            None,
        )
        .next()
        .await;

    let create_options = CreateContainerOptions {
        name: Some(create_conf.container_name.clone()),
        platform: "linux".to_string(),
    };
    // Randomly assign a port
    let host_config: HostConfig = HostConfig {
        extra_hosts: Some(vec!["host.docker.internal:host-gateway".into()]),
        port_bindings: Some(hashmap! {
            format!("{}/tcp", create_conf.internal_port) => None,
        }),
        device_requests: Some(vec![DeviceRequest {
            driver: Some("".into()),
            count: Some(-1),
            device_ids: None,
            capabilities: Some(vec![vec!["gpu".into()]]),
            options: Some(HashMap::new()),
        }]),
        ..Default::default()
    };

    let mut exposed_ports = Vec::new();
    let exposed_port = format!("{}/tcp", create_conf.internal_port.to_string());
    exposed_ports.push(exposed_port);

    let config = ContainerCreateBody {
        image: Some(create_conf.image_name),
        cmd: None,
        exposed_ports: Some(exposed_ports),
        host_config: Some(host_config),
        ..Default::default()
    };

    let _container = daemon.create_container(Some(create_options), config).await?;

    find_container(daemon, &create_conf.container_name)
        .await?
        .context("Failed to create container")
}

pub async fn find_container(daemon: &Docker, name: &str) -> eyre::Result<Option<ContainerSummary>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    map.insert("name".to_string(), vec![name.to_string()]);
    let opts = ListContainersOptions {
        all: true,
        limit: None,
        size: false,
        filters: Some(map),
    };

    Ok(daemon.list_containers(Some(opts)).await?.into_iter().next())
}
