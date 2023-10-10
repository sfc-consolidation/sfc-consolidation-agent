# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Creates, updates, and deletes a job object.
"""

import json
from time import sleep
from typing import Union

from kubernetes import client, config

from app.types import State

JOB_NAME = "simulator"
NAMESPACE = "sfc-consolidation"


def create_job_object(id: str, state: Union[State, None] = None):
    # Configure Pod Command
    command=[
            "java",
            "-jar",
            "SFC_Consolidation_Simulator-0.0.1.jar",
            "-m",
            "STEP",
            "--id",
            id, 
        ]
    if state != None:
        command = command + [
            "-s",
            json.dumps(state, default=lambda o: o.__dict__),
        ]
    # Configure Pod template container
    container = client.V1Container(
        name="simulator",
        image="docker.io/library/sfc-consolidation-simulator:0.0.1",
        image_pull_policy="Never",
        command=command,
    )
    toleration = client.V1Toleration(
        key="node-role.kubernetes.io/control-plane",
        operator="Exists",
        effect="NoSchedule",
    )
    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            name=f"{JOB_NAME}-{id}",
            labels={"jobgroup": JOB_NAME},
        ),
        spec=client.V1PodSpec(
            containers=[container],
            restart_policy="OnFailure",
            host_network=True,
            tolerations=[toleration]
        )
    )
    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template
    )
    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=f"{JOB_NAME}-{id}",
            labels={"jobgroup": JOB_NAME},
            namespace=NAMESPACE,
        ),
        spec=spec)

    return job


def create_job(api_instance, job, id: str):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace=NAMESPACE)
    print(f"Job created. status='{str(api_response.status)}'")
    # get_job_status(api_instance, id)


def get_job_status(api_instance, id: str):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=f"{JOB_NAME}-{id}",
            namespace=NAMESPACE)
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
        sleep(1)
        print(f"Job status='{str(api_response.status)}'")

def wait_job_complete(api_instance, id: str):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=f"{JOB_NAME}-{id}",
            namespace=NAMESPACE)
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
    print(f"{JOB_NAME}-{id} is completed.")



def delete_job(api_instance, id):
    api_response = api_instance.delete_namespaced_job(
        name=f"{JOB_NAME}-{id}",
        namespace=NAMESPACE,
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    print(f"Job deleted. status='{str(api_response.status)}'")


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    

    for id in range(10):
        id = str(id)
        job = create_job_object(id)
        create_job(batch_v1, job, id)

    for id in range(10):
        wait_job_complete(batch_v1, id)
        delete_job(batch_v1, id)


if __name__ == '__main__':
    main()