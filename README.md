# Clearboard – Stream an enhanced blackboard 👨‍🏫

Clearboard receives a stream of images, detects a blackboard in it, extracts it and serves
an enhanced view of what is written on the blackboard.

## Getting started

### Prerequisite

Make sure you have a recent version of Docker and
[Docker Compose](https://docs.docker.com/compose/install) installed on your laptop:

```bash
$ docker -v
  Docker version 20.10.2, build 2291f61

$ docker-compose -v
  docker-compose version 1.27.4, build 40524192
```

>⚠️ You may need to run the following commands with `sudo` but this can be
>avoided by assigning your user to the `docker` group.

### Project bootstrap

The easiest way to start working on the project is to use our `Makefile` :
```bash
$ make bootstrap
```

This command builds the `app` and `lambda-enhance` containers, installs
dependencies and runs them. It's a good idea to use this command each time
you are pulling code from the project repository to avoid dependency-releated
issues.

#### FastAPI app

The `app` container is the FastAPI web server that serves the API to:

- serve an upload policy to the frontend
- serve a CloudFront signed url to retrieve the processed image from the destination bucket

You should be able to access the API overview interface at (http://localhost:8070).

#### Lambda enhance

The `lambda-enhance` container holds the Python script that processes images in an AWS lambda.

The lambda is triggered by S3 each time an image is uploaded to it. It processes the image and
deposits the result in the destination bucket.

The lambda container can run locally in docker compose for development purposes:

```bash
$ make lambda
```

You can then test it with a trigger request including a payload mimicking an AWS S3 trigger.
For example, if your source bucket contains a file with key "image.png", the following
curl mimicks the trigger sent by S3 when the file was uploaded:

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
    -d '{"Records":[{"s3":{"bucket":{"name": "production-clearboard-source-fun"}, "object":{"key": "image.png"}}}]}'
```

Note that you can see all available commands in our `Makefile` with :

```bash
$ make help
```

## Guides

## Contributing

This project is intended to be community-driven, so please, do not hesitate to
get in touch if you have any question related to our implementation or design
decisions.

We try to raise our code quality standards and expect contributors to follow
the recommandations from our
[handbook](https://openfun.gitbooks.io/handbook/content).

## License

This work is released under the MIT License (see [LICENSE](./LICENSE)).
