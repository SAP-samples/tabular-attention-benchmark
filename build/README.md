# Build

Docker image build scripts for the tabular attention benchmark.

## Prerequisites: Running Docker as your user

By default, Docker requires `sudo`. To run Docker commands as `<username>`:

1. Add your user to the `docker` group:

   ```bash
   sudo usermod -aG docker <username>
   ```

2. Apply the group change (either log out and back in, or run for the current shell):

   ```bash
   newgrp docker
   ```

3. Verify:

   ```bash
   docker run hello-world
   ```

> **Note:** Members of the `docker` group have root-equivalent privileges on the host. This is fine for a personal dev machine but should be considered in shared environments.
