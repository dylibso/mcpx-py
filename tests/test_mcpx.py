from mcpx_py import Client

import unittest


class TestClient(unittest.TestCase):
    def client(self):
        try:
            client = Client()
            return client
        except Exception:
            return None

    def test_list_installs(self):
        client = self.client()
        if client is None:
            return
        for v in client.installs.values():
            self.assertTrue(v.name != "")

    def test_search(self):
        client = self.client()
        if client is None:
            return
        res = client.search("fetch")
        self.assertEqual(res[0]["slug"], "bhelx/fetch")


if __name__ == "__main__":
    unittest.main()
