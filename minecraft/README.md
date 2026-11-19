# Minecraft deployments

This directory contains three Kubernetes deployments. Each directory has a `deploy.sh` script and a setup guide. Each variant uses its own namespace.

## Variants

| Directory | Server path | Namespace | Client version |
|---|---|---|---|
| [`java/`](java/README.md) | Java server behind an nginx TCP proxy | `minecraft-java` | Java Edition, Latest Release by default |
| [`java-no-nginx/`](java-no-nginx/README.md) | Java server directly behind its LoadBalancer | `minecraft-java-no-nginx` | Java Edition, Latest Release by default |
| [`cubernite/`](cubernite/README.md) | Cuberite server directly behind its LoadBalancer | `minecraft-cubernite` | Java Edition 1.12.2 recommended |

### Java with nginx

`java/` uses the `itzg/minecraft-server:latest` image. The public Service sends traffic to nginx. nginx sends the traffic to the Java server through its headless Service.

### Java without nginx

`java-no-nginx/` uses the same Java server image. Its public Service sends traffic directly to the game pod. It has no nginx resources or Ingress. Its separate resource names let it run alongside the nginx variant.

### Cuberite

`cubernite/` uses the alternative C++ Cuberite server, with image `lkup77/cuberite:v3`. It has no nginx resources. Use a supported legacy Java Edition client. The supplied guide recommends version 1.12.2.

## Deploy and connect

Open the guide for the chosen variant. Run its `deploy.sh` script from that directory. The script creates the namespace and deploys that variant.

Use the guide's `kubectl get service` command to obtain the public address. Add `<EXTERNAL-IP>:25565` under **Multiplayer → Add Server** in the supported client.

## Notes

- **Node selectors:** Adjust the game Deployments to match your cluster before deployment. Game pods currently require `live.cast.ai/migration-enabled=true`. Change the `nodeSelector` in each game's `deployment.yaml` if your nodes use different labels. Nginx has no node selector.
- The Java `VERSION=LATEST` setting can change the server version on an image update. Each Java guide explains version pinning.
- All three variants disable online account verification and expose public LoadBalancers. Restrict access before use beyond testing.
- These configurations have no persistent volumes for world data. Replacing game pods removes their local world data.
- Both Java directories contain `placeholder-small.yaml`, which defines a separate test workload. The deployment scripts do not apply it.
- Cuberite's web admin listens on port 8080. The supplied Service exposes only game port 25565.
