import subprocess


def docker():

    process = subprocess.run(["which", "podman"], stdout=subprocess.DEVNULL)
    if (process.returncode == 0):
        return "podman"

    process = subprocess.run(["which", "docker"], stdout=subprocess.DEVNULL)
    if (process.returncode == 0):
        return "docker"

    return "sudo docker"
