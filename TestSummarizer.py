import unittest
from Summarizer import Summarizer
import json
from unittest.mock import patch, MagicMock


class TestSummarizer(unittest.TestCase):
    def setUp(self):
        """Set up a Summarizer instance before each test."""
        self.summarizer = Summarizer()
        self.test_text = "This is a test sentence. " * 100  # Create some dummy text


    def test_count_tokens(self):
        """Test the _count_tokens method with different inputs."""
        test_cases = [
            ("Hello world", 2),  # Simple case
            ("", 0),
            ("Hello   world", 3),  # Multiple spaces
            ("This is a longer sentence with more words.", 9)
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.summarizer._count_tokens(text)
                self.assertEqual(result, expected)


    def test_split_text_into_chunks(self):
        """Test the _split_text_into_chunks method."""
        # Create a text that should split into exactly 2 chunks
        test_text = "word " * 1000
        max_tokens = 500

        chunks = self.summarizer._split_text_into_chunks(test_text, max_tokens)

        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks
        for chunk in chunks:
            tokens = self.summarizer._count_tokens(chunk)
            self.assertLessEqual(tokens, max_tokens)  # Each chunk should be within limit


    @patch('requests.post')
    def test_process_chunk(self, mock_post):
        """Test the _process_chunk method with mocked API response."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'chunk_summary': 'Test summary',
            'updated_context': 'Test context'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        chunk_text = "Test chunk"
        context_summary = "Previous context"

        chunk_summary, updated_context = self.summarizer.process_chunk(
            chunk_text, context_summary
        )

        self.assertEqual(chunk_summary, 'Test summary')
        self.assertEqual(updated_context, 'Test context')

        # Test error handling
        mock_post.side_effect = Exception("API Error")
        chunk_summary, updated_context = self.summarizer.process_chunk(
            chunk_text, context_summary
        )
        self.assertEqual(chunk_summary, '')
        self.assertEqual(updated_context, context_summary)


    @patch('requests.post')
    def test_summarize_text(self, mock_post):
        """Test the _summarize_text method."""
        mock_response = MagicMock()
        mock_response.text = "Summarized text"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.summarizer._summarize_text("Test text", 100)
        self.assertEqual(result, "Summarized text")     # Summarize: text text text text into "Summarized text"


    def test_reduce_history(self):
        """Test the _reduce_history method."""
        # Create a history list that exceeds the threshold
        long_history = ["Long summary text" * 100] * 10

        reduced_history = self.summarizer._reduce_history(long_history)

        combined_tokens = self.summarizer._count_tokens(" ".join(reduced_history))
        self.assertLessEqual(
            combined_tokens,
            self.summarizer.FINAL_SUMMARY_THRESHOLD
        )


    @patch('requests.post')
    def test_final_summary(self, mock_post):
        """Test the _final_summary method."""
        mock_response = MagicMock()
        mock_response.text = "Final summary"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        history_list = ["Summary 1", "Summary 2"]
        result = self.summarizer._final_summary(history_list)
        self.assertEqual(result, "Final summary")


    @patch('requests.post')  # Add this to mock all API calls
    @patch.object(Summarizer, '_process_chunk')
    def test_summarize_full_flow(self, mock_process, mock_post):  # Add mock_post parameter
        """Test the complete summarize method flow."""
        # Mock the process_chunk method to return predictable results
        mock_process.return_value = ("Chunk summary", "Updated context")

        # Mock the API response for final summary
        mock_response = MagicMock()
        mock_response.text = "Final summary text"
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Test input
        test_input = "Test sentence. " * 1000

        # Run the summarize method
        result = self.summarizer.summarize(test_input)

        # Verifying the result is not empty
        self.assertTrue(result)
        # Verifying process_chunk was called at least 1
        self.assertTrue(mock_process.called)
        # Verifying we got the mocked final summary
        self.assertEqual(result, "Final summary text")


if __name__ == '__main__':
    unittest.main()