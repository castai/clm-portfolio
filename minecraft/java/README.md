# Minecraft Java with nginx

Java Edition server behind an nginx TCP proxy. All resources use the `minecraft-java` namespace.

## Deploy

From the repository root:

```sh
cd minecraft/java
kubectl config current-context  # Confirm this is the intended cluster.
./deploy.sh
```

The script creates the namespace, applies the manifests, and waits for the rollout. Configure `kubectl` for your cluster before deployment.

## Minecraft version

Use **Minecraft Java Edition, Latest Release**. The `deployment.yaml` file sets `TYPE=VANILLA` and `VERSION=LATEST`.

To use a fixed version, change `VERSION` in `deployment.yaml` to the desired Java Edition release. Rerun `./deploy.sh`. Select that release in the Minecraft Launcher.

## Connect

Get the public address:

```sh
kubectl get service minecraft-java -n minecraft-java
```

Wait until `EXTERNAL-IP` shows an IP address or hostname rather than `<pending>`. In Minecraft, open **Multiplayer → Add Server** and enter:

```text
<EXTERNAL-IP>:25565
```

If `EXTERNAL-IP` is a hostname, use that hostname.

**Note:** These settings have `ONLINE_MODE=FALSE` and expose a public LoadBalancer. The server does not verify Minecraft account identities. Restrict access before use beyond testing.
