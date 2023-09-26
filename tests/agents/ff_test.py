import unittest
from app.types import State, SRV, Rack, VNF
from app.agents.ff import FFAgent


class TestFF(unittest.TestCase):

    def setUp(self):
        self.agent = FFAgent

    def test_stops_when_no_server_to_allocate_vnf(self):
        # Given: a state where no server can allocate the VNF
        servers = [SRV(id=0, totVcpuNum=2, totVmemMb=1000, sleepable=False)]
        racks = [Rack(id=0, srvList=servers)]
        vnfs = [VNF(id=0, srvId=0, sfcId=0, orderInSfc=1,
                    reqVcpuNum=3, reqVmemMb=1500, movable=True)]
        # Adjusted based on your type definition
        state = State(rackList=racks, sfcList=[], vnfList=vnfs)

        # When: the inference method is called
        action = self.agent.inference(state)

        # Then: FF agent should stop (assuming None signifies a "stop" action)
        self.assertIsNone(action)

    def test_stops_when_no_vnf_to_allocate(self):
        # Given: a state where there is no VNF to allocate
        state = State(rackList=[], sfcList=[], vnfList=[])

        # When: the inference method is called
        action = self.agent.inference(state)

        # Then: FF agent should stop
        self.assertIsNone(action)

    def test_stops_when_one_vnf_but_not_movable(self):
        # Given: a state where there is one VNF but it is not movable
        servers = [SRV(id=0, totVcpuNum=2, totVmemMb=1000, sleepable=False), SRV(
            id=1, totVcpuNum=2, totVmemMb=1000, sleepable=False)]
        racks = [Rack(id=1, srvList=servers)]
        vnfs = [VNF(id=0, srvId=0, sfcId=1, orderInSfc=1,
                    reqVcpuNum=2, reqVmemMb=1000, movable=False)]
        state = State(rackList=racks, sfcList=[], vnfList=vnfs)

        # When: the inference method is called
        action = self.agent.inference(state)

        # Then: FF agent should stop
        self.assertIsNone(action)

    def test_normal_example(self):
        # Given: a state where there is two servers and three VNFs,
        #       and the each server has one and two VNFs.
        #       Each VNF has the same resource usage, Server, too.
        servers = [SRV(id=0, totVcpuNum=4, totVmemMb=2000, sleepable=False), SRV(
            id=1, totVcpuNum=2, totVmemMb=1000, sleepable=False)]
        racks = [Rack(id=0, srvList=servers)]
        vnfs = [VNF(id=0, srvId=0, sfcId=1, orderInSfc=1,
                    reqVcpuNum=1, reqVmemMb=500, movable=True),
                VNF(id=1, srvId=0, sfcId=1, orderInSfc=2,
                    reqVcpuNum=1, reqVmemMb=500, movable=True),
                VNF(id=2, srvId=1, sfcId=1, orderInSfc=3,
                    reqVcpuNum=1, reqVmemMb=500, movable=True)]
        state = State(rackList=racks, sfcList=[], vnfList=vnfs)

        # When: the inference method is called.
        action = self.agent.inference(state)

        # Then: FF agent should select the VNF in the lower resource server to higher resource server.
        self.assertEqual(action.vnfId, 2)
        self.assertEqual(action.srvId, 0)


if __name__ == "__main__":
    unittest.main()
