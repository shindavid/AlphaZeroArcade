# AlphaZeroArcade


## Docker Installation

Once docker is installed, build the image by running the following from the project root directory

```bash
docker build -t alphazeroarcade .
```

Docker-compose is used to run the app. For example you can run the image with:
```bash
docker compose up
```

If you want to interact with the bash shell in the image, run
```bash
docker compose up -d
docker-compose exec server bash
```

By default the repo is mounted to the `/AlphaZeroArcade` directory within the image
