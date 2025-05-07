import unittest
import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

from mcpx_py.chat import Chat


class TestChat(unittest.TestCase):
    """Tests for the Chat class"""

    def setUp(self):
        """Setup for each test"""
        # Create a mocked Agent for testing without API calls
        self.mock_agent = MagicMock()
        self.mock_agent.run = AsyncMock(return_value="Mocked response")
        
        # Mock run_mcp_servers context manager
        mock_servers = MagicMock()
        mock_servers.__aenter__ = AsyncMock()
        mock_servers.__aexit__ = AsyncMock()
        self.mock_agent.run_mcp_servers.return_value = mock_servers
        
        # Setup mock for iteration
        self.mock_node = MagicMock()
        self.mock_node.response = "Response content"
        
        # Create a proper async iterator mock
        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)
        
        # Create context manager for iter
        mock_iter_cm = MagicMock()
        mock_iter_cm.__aenter__ = AsyncMock()
        mock_iter_cm.__aexit__ = AsyncMock()
        
        # Set up the iterator to return our mock node
        mock_iter_cm.__aenter__.return_value = AsyncIterator([self.mock_node])
        self.mock_agent.iter.return_value = mock_iter_cm

    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_chat_init(self, mock_capture, mock_agent_class):
        """Test Chat initialization"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        
        # Act
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Assert
        self.assertIsNotNone(chat.agent)
        self.assertEqual(chat.history, [])
        mock_agent_class.assert_called_once()
        
    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_chat_clear_history(self, mock_capture, mock_agent_class):
        """Test clearing chat history"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        chat = Chat(model="claude-3-haiku-20240307")
        chat.history = ["message1", "message2"]
        
        # Act
        chat.clear_history()
        
        # Assert
        self.assertEqual(chat.history, [])

    def run_async(self, coro):
        """Helper method to run async code in tests"""
        # Filter out the RuntimeWarning about coroutines
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine.*was never awaited")
            return asyncio.run(coro)
            
    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_send_message(self, mock_capture, mock_agent_class):
        """Test sending a message"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=["captured_message"])
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_capture.return_value = mock_context
        
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Act - Run the async function in the test
        result = self.run_async(chat.send_message("Hello, AI!"))
        
        # Assert
        self.assertEqual(result, "Mocked response")
        self.mock_agent.run.assert_called_once()
        self.assertEqual(chat.history, ["captured_message"])

    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    @patch('asyncio.new_event_loop')
    def test_send_message_sync(self, mock_new_loop, mock_capture, mock_agent_class):
        """Test sending a message synchronously"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=["captured_message"])
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_capture.return_value = mock_context
        
        # Setup mock loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock(return_value="Mocked response")
        mock_new_loop.return_value = mock_loop
        
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Act
        result = chat.send_message_sync("Hello, AI!")
        
        # Assert
        mock_loop.run_until_complete.assert_called_once()
        mock_loop.close.assert_called_once()
        self.assertEqual(result, "Mocked response")

    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_iter(self, mock_capture, mock_agent_class):
        """Test iterating through responses"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=["captured_message"])
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_capture.return_value = mock_context
        
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Act & Assert - Run the async test
        async def test_async():
            nodes = []
            async for node in chat.iter("Hello, AI!"):
                nodes.append(node)
            return nodes
            
        nodes = self.run_async(test_async())
        
        # Assert
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0], self.mock_node)

    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_iter_content(self, mock_capture, mock_agent_class):
        """Test iterating through content"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=["captured_message"])
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_capture.return_value = mock_context
        
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Act & Assert - Run the async test
        async def test_async():
            contents = []
            async for content in chat.iter_content("Hello, AI!"):
                contents.append(content)
            return contents
            
        contents = self.run_async(test_async())
        
        # Assert
        self.assertEqual(contents, ["Response content"])

    @patch('mcpx_py.chat.Agent')
    @patch('mcpx_py.chat.pydantic_ai.capture_run_messages')
    def test_inspect(self, mock_capture, mock_agent_class):
        """Test inspecting a message exchange"""
        # Arrange
        mock_agent_class.return_value = self.mock_agent
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=["captured_message"])
        mock_context.__exit__ = MagicMock(return_value=None)
        mock_capture.return_value = mock_context
        
        chat = Chat(model="claude-3-haiku-20240307")
        
        # Act
        result, messages = self.run_async(chat.inspect("Hello, AI!"))
        
        # Assert
        self.assertEqual(result, "Mocked response")
        self.assertEqual(messages, ["captured_message"])

if __name__ == '__main__':
    unittest.main()
