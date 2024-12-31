from mcpx import Client

import unittest


class TestClient(unittest.TestCase):

    def test_list_installs(self):
        client = Client()
        for v in client.installs.values():
            self.assertTrue(v.name != '')


if __name__ == '__main__':
    unittest.main()
