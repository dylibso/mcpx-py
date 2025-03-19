import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from mcpx_py.chat import Chat
from mcpx_py.builtin_tools import TOOLS

import logging

logger = logging.getLogger(__name__)


class AsyncMockContext:
    """Mock for async context managers"""

    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.index = 0
        self.run = self  # Make self act as the run object
        logger.info(f"AsyncMockContext created with {len(self.nodes)} nodes")

    async def __aenter__(self):
        logger.info("AsyncMockContext.__aenter__ called")
        return self.run

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("AsyncMockContext.__aexit__ called")
        return None

    def __aiter__(self):
        logger.info("AsyncMockContext.__aiter__ called")
        return self

    async def __anext__(self):
        logger.info(
            f"AsyncMockContext.__anext__ called, index={self.index}, len={len(self.nodes)}"
        )
        if self.index >= len(self.nodes):
            logger.info("AsyncMockContext: raising StopAsyncIteration")
            raise StopAsyncIteration
        node = self.nodes[self.index]
        self.index += 1
        logger.info(f"AsyncMockContext: yielding node {node}")
        return node


class TestMcpx(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Setup test with mocked dependencies"""
        self.maxDiff = None

        # Mock the config module to prevent session ID errors - do this before any initialization
        patcher = patch("mcp_run.config._default_session_id", return_value="test-session")
        self.mock_session = patcher.start()
        self.addCleanup(patcher.stop)

        # Mock the list_installs response to prevent unauthorized errors
        # Create a mock client with empty tools/installs
        mock_client = MagicMock()
        mock_client.tools = {}
        mock_client.installs = []
        mock_client.list_installs = MagicMock(return_value=[])

        # Path the Client class to return our mock
        patcher = patch("mcpx_pydantic_ai.mcp_run.Client", return_value=mock_client)
        self.mock_client = patcher.start()
        self.addCleanup(patcher.stop)

        # Create chat instance after mocks are in place
        with patch("mcpx_py.chat.pydantic_ai", autospec=True) as mock_pydantic_ai:
            # Configure the mock for capturing run messages
            mock_messages = []
            mock_messages_ctx = MagicMock()
            mock_messages_ctx.__enter__ = MagicMock(return_value=mock_messages)
            mock_messages_ctx.__exit__ = MagicMock(return_value=None)
            mock_pydantic_ai.capture_run_messages.return_value = mock_messages_ctx

            # Create chat instance once mocks are in place
            self.chat = Chat()

            # Create agent mock with proper coroutine support
            self.mock_agent = AsyncMock()

            # Configure register_tool to work synchronously
            register_tool_mock = MagicMock()
            self.mock_agent.register_tool = register_tool_mock

            # Configure set_profile to work synchronously
            set_profile_mock = MagicMock()
            self.mock_agent.set_profile = set_profile_mock

            # Setup iter support
            self.mock_agent._iter_nodes = []

            def iter_side_effect(*args, **kwargs):
                return AsyncMockContext(self.mock_agent._iter_nodes)

            self.mock_agent.iter = MagicMock(side_effect=iter_side_effect)

            # Replace the chat's agent with our mock
            self.chat.agent = self.mock_agent

    def test_initialization(self):
        """Test chat initialization and builtin tool registration"""
        # Register tools now for verification
        self.chat._register_builtins()

        # Verify builtin tools are registered
        for tool in TOOLS:
            self.mock_agent.register_tool.assert_any_call(
                tool, getattr(self.chat, "_tool_" + tool.name)
            )

    def test_history_management(self):
        """Test chat history management"""
        # Initially empty
        self.assertEqual(len(self.chat.history), 0)

        # Clear history
        self.chat.history = ["test message"]
        self.chat.clear_history()
        self.assertEqual(len(self.chat.history), 0)

    async def test_send_message(self):
        """Test sending messages"""
        test_msg = "Hello"
        expected_response = "Response"

        # Configure async mock return
        self.mock_agent.run.return_value = expected_response

        # Configure mock agent to update history
        self.chat.history = []

        # Test async send_message
        response = await self.chat.send_message(test_msg)

        # Verify agent.run was called with correct params
        self.mock_agent.run.assert_called_with(
            test_msg,
            message_history=[],
        )

        # Verify response
        self.assertEqual(response, expected_response)

    def test_send_message_sync(self):
        """Test sending synchronous messages"""
        test_msg = "Hello"
        expected_response = "Response"

        # Create a synchronous mock method
        self.mock_agent.run_sync = MagicMock()
        self.mock_agent.run_sync.return_value = expected_response

        # Configure mock agent to update history
        self.chat.history = []

        # Test sync send_message
        response = self.chat.send_message_sync(test_msg)

        # Verify agent.run_sync was called with correct params
        self.mock_agent.run_sync.assert_called_with(
            test_msg,
            message_history=[],
        )

        # Verify response
        self.assertEqual(response, expected_response)

    # async def test_mcp_run_search_servlets(self):
    #     """Test servlet search tool"""
    #     # Mock client search response
    #     mock_result = MagicMock()
    #     mock_result.slug = "test/servlet"
    #     mock_result.meta = {"name": "Test", "description": "Test servlet"}
    #     mock_result.installation_count = 10

    #     # Create a proper async iterator for search results
    #     async def async_search_iter():
    #         yield mock_result

    #     self.mock_agent.client.search = AsyncMock(return_value=[mock_result])

    #     # Test search
    #     result = self.chat._tool_mcp_run_search_servlets({"q": "test"})

    #     # Verify client search was called
    #     self.mock_agent.client.search.assert_called_with("test")

    #     # Verify response format
    #     self.assertEqual(len(result), 1)
    #     self.assertEqual(result[0]["slug"], "test/servlet")
    #     self.assertEqual(result[0]["schema"]["name"], "Test")
    #     self.assertEqual(result[0]["installation_count"], 10)

    #     # Test empty query
    #     result = await self.chat._tool_mcp_run_search_servlets({"q": ""})
    #     self.assertEqual(result, "ERROR: provide a query when searching")

    def test_mcp_run_get_profiles(self):
        """Test profile listing tool"""
        # Create mock profile
        mock_profile = MagicMock()
        mock_profile.slug = "profile1"
        mock_profile.description = "Test profile 1"

        # Mock profiles dictionary
        mock_profiles = {"user1": {"profile1": mock_profile}}
        self.mock_agent.client.profiles = mock_profiles

        # Test get profiles
        result = self.chat._tool_mcp_run_get_profiles({})

        # Verify response format
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "user1/profile1")
        self.assertEqual(result[0]["description"], "Test profile 1")

    def test_mcp_run_set_profile(self):
        """Test profile setting tool"""
        # Test with full profile name
        result = self.chat._tool_mcp_run_set_profile({"profile": "user/profile"})
        self.mock_agent.set_profile.assert_called_with("user/profile")
        self.assertEqual(result, "Active profile set to user/profile")

        # Test with short profile name
        result = self.chat._tool_mcp_run_set_profile({"profile": "profile"})
        self.mock_agent.set_profile.assert_called_with("~/profile")
        self.assertEqual(result, "Active profile set to ~/profile")

    async def test_iter(self):
        """Test iterator functionality"""
        test_msg = "Hello"
        mock_node1 = MagicMock(response="Part 1")
        mock_node2 = MagicMock(response="Part 2")

        # Set up what the iterator should yield
        logger.info("test_iter: setting up mock nodes")
        self.mock_agent._iter_nodes = [mock_node1, mock_node2]

        # Test the iterator
        nodes = []
        logger.info("test_iter: starting iteration")
        async for node in self.chat.iter(test_msg):
            logger.info(f"test_iter: got node {node}")
            nodes.append(node)
        logger.info("test_iter: finished iteration")

        # Verify agent.iter was called with correct params
        self.mock_agent.iter.assert_called_with(
            test_msg,
            message_history=[],
        )

        # Verify nodes were yielded
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0], mock_node1)
        self.assertEqual(nodes[1], mock_node2)

    async def test_iter_content(self):
        """Test content iterator functionality"""
        test_msg = "Hello"

        # Create a mock node class with the attributes we need
        class MockNode:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # Create mock nodes with different attribute types
        nodes = [
            MockNode(response="Response content"),
            MockNode(model_response="Model response content"),
            MockNode(request="Request content"),
            MockNode(model_request="Model request content"),
            MockNode(data="Data content"),
        ]

        # Set up what the iterator should yield
        logger.info("test_iter_content: setting up mock nodes")
        self.mock_agent._iter_nodes = nodes

        # Test the content iterator
        contents = []
        logger.info("test_iter_content: starting iteration")
        async for content in self.chat.iter_content(test_msg):
            logger.info(f"test_iter_content: got content {content}")
            contents.append(content)
        logger.info("test_iter_content: finished iteration")

        # Verify agent.iter was called with correct params
        self.mock_agent.iter.assert_called_with(
            test_msg,
            message_history=[],
        )

        # Verify correct content was extracted from each node type
        self.assertEqual(len(contents), 5)
        self.assertEqual(contents[0], "Response content")
        self.assertEqual(contents[1], "Model response content")
        self.assertEqual(contents[2], "Request content")
        self.assertEqual(contents[3], "Model request content")
        self.assertEqual(contents[4], "Data content")

    async def test_inspect(self):
        """Test inspect functionality"""
        test_msg = "Hello"
        expected_response = "Response"
        expected_messages = []

        # Configure async mock return
        self.mock_agent.run.return_value = expected_response

        # Test inspect functionality
        response = await self.chat.inspect(test_msg)

        # Verify agent.run was called with correct params
        self.mock_agent.run.assert_called_with(
            test_msg,
            message_history=[],
        )

        # Verify response
        self.assertEqual(response, (expected_response, expected_messages))


if __name__ == "__main__":
    unittest.main()
