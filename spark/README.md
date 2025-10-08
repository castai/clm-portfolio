These are examples of SparkApplications used for live migration testing.

The spark-pi example can be deployed as is, but for the streaming benchmark example you need to build and push an image.

## Build

```sh
docker build . --platform linux/amd64 -f Dockerfile-benchmark -t <your-image-name-and-tag>
docker push <your-image-name-and-tag>
```

After building don't forget to update the image in the `spark-streaming-benchmark.yaml`
