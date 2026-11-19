# Cuberite

Cuberite is a Minecraft-compatible server. The `lkup77/cuberite:v3` image serves the game on TCP port 25565 in the `minecraft-cubernite` namespace.

## Deploy

From the repository root:

```sh
cd minecraft/cubernite
kubectl config current-context  # Confirm this is the intended cluster.
./deploy.sh
```

The script creates the namespace. It applies the ConfigMap, Deployment, and Service. It waits for Cuberite to become ready. Configure `kubectl` for your cluster before deployment.

## Minecraft version

Use **Minecraft Java Edition 1.12.2**. Cuberite supports Java Edition client protocols **1.8 through 1.12.2**, as documented on [the official Cuberite site](https://cuberite.org/). This server does not support Bedrock Edition clients.

In the Minecraft Launcher, open **Installations → New installation**, select **release 1.12.2**, and start that installation. The client selects the version. This Deployment has no `VERSION` variable.

## Connect

Get the public address:

```sh
kubectl get service cuberite -n minecraft-cubernite
```

Wait until `EXTERNAL-IP` shows an IP address or hostname rather than `<pending>`. In Minecraft, open **Multiplayer → Add Server** and enter:

```text
<EXTERNAL-IP>:25565
```

**Note:** The supplied settings disable authentication (`Authenticate=0`) and expose a public LoadBalancer. The server does not verify Minecraft account identities. Restrict access before use beyond testing.
