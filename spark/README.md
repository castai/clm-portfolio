These are examples of SparkApplications used for live migration testing.

The spark-pi example can be deployed as is, but for the streaming benchmark example you need to build and push an image.

## Build

```sh
docker build . --platform linux/amd64 -f Dockerfile-benchmark -t <your-image-name-and-tag>
docker push <your-image-name-and-tag>
```

After building don't forget to update the image in the `spark-streaming-benchmark.yaml`

## Deploy

Install spark:
```sh
helm install spark-operator spark-operator/spark-operator \
    --namespace spark-operator \
    --create-namespace
```

Give spark permissions:
```sh
kubectl apply -f spark-accounts.yaml 
```

Deploy the spark application:
```sh
kubectl apply -f spark-streaming-benchmark.yaml
```
