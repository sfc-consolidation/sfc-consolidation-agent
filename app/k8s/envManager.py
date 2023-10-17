import socket
import requests
from time import sleep

from kubernetes import client, config
from app.trainers.env import Environment, ResetArg, AsyncEnvironment
from app.types import Action

POD_NAME_PREFIX = "simulator-api"
NAMESPACE = "sfc-consolidation"


class EnvManager:
    def __init__(self):
        config.load_kube_config()
        api = client.CoreV1Api()
        self.start_port = 9000
        self.ids = []
        self.api: client.CoreV1Api = api
    
    def get_pod_name(self, id: str):
        return f"{POD_NAME_PREFIX}-{id}"

    def create_pod_object(self, id: str, port: int):
        # Instantiate the pod object
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(
                name=self.get_pod_name(id),
                namespace=NAMESPACE,
            ),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="simulator",
                        image="docker.io/library/sfc-consolidation-simulator-api:0.0.1",
                        image_pull_policy="Never",
                        command=[
                            "java",
                            "-jar",
                            "SFC_Consolidation_Simulator_API-0.0.1.jar",
                            f"--server.port={port}",
                        ],
                        ports=[
                            client.V1ContainerPort(port)
                        ],
                    ),
                ],
                restart_policy="OnFailure",
                host_network=True,
                tolerations=[client.V1Toleration(
                    key="node-role.kubernetes.io/control-plane",
                    operator="Exists",
                    effect="NoSchedule",
                )],
            ),
        )
        return pod

    def wait_pod_ready(self, id: str):
        status_code = -1    
        while status_code != 200:
            sleep(1)
            try:
                response = requests.get(f"http://{self.find_target_by_id(id)}/healthy")
                status_code = response.status_code
                print(response.text)
            except requests.exceptions.RequestException as e:
                print(f"Healthy Check Fail: {e}")
        return True

    def find_target_by_id(self, id: str):
        pod_name = self.get_pod_name(id)
        
        pod_info = self.api.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
        port = pod_info.spec.containers[0].ports[0].container_port
        
        node_name = pod_info.spec.node_name

        ip = self.api.read_node(name=node_name).status.addresses[0].address

        return f'{ip}:{port}'

    def check_target_usage(self, ip: str, port: int):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip, port))
        sock.close()
        if result == 0:
            return True
        else:
            return False

    def get_all_ips(self):
        ips = []
        nodes = self.api.list_node()
        for node in nodes.items:
            ips.append(node.status.addresses[0].address)
        return ips

    def create_env(self, id: str, is_async: bool = False):
        port = self.start_port
        is_usage = True
        ips = self.get_all_ips()
        while is_usage:
            is_usage = False
            for ip in ips:
                if self.check_target_usage(ip, port):
                    is_usage = True
                    break
            if is_usage:
                port += 1
        self.start_port = port + 1

        pod_obj = self.create_pod_object(id, port)
        self.api.create_namespaced_pod(namespace=NAMESPACE, body=pod_obj)
        self.ids.append(id)
        self.wait_pod_ready(id)
        if is_async:
            return AsyncEnvironment(lambda: self.find_target_by_id(id))
        return Environment(lambda: self.find_target_by_id(id))


    def delete_all(self):
        for id in self.ids:
            pod_name = self.get_pod_name(id)
            self.api.delete_namespaced_pod(namespace=NAMESPACE, name=pod_name)


def example0():
    envManager = EnvManager()
    for i in range(10):
        id = f"test{i}"
        env = envManager.create_env(id)
        print(env.get_target_address_fn())
    envManager.delete_all()

def example1():
    try:
        envManager = EnvManager()
        id = "test0"
        env = envManager.create_env(id)
        resetArg = ResetArg(
            maxRackNum=2, minRackNum=2,
            maxSrvNumInSingleRack=3, minSrvNumInSingleRack=3,
            maxVnfNum=10, minVnfNum=10,
            maxSfcNum=3, minSfcNum=3,
            maxSrvVcpuNum=100, minSrvVcpuNum=100,
            maxSrvVmemMb=32 * 1024, minSrvVmemMb=32 * 1024,
            maxVnfVcpuNum=10, minVnfVcpuNum=1,
            maxVnfVmemMb=1024 / 2, minVnfVmemMb=1024 * 4,
        )
        state, info, done = env.reset(resetArg)
        while not done:
            print(info)
            action = Action(vnfId=0, srvId=0)
            state, info, done = env.step(action)
        print(info)
    finally:
        envManager.delete_all()

if __name__ == "__main__":
    example1()