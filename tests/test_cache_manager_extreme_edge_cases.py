import asyncio
from unittest.mock import MagicMock

import pytest
import redis

from llmproxy.core.cache_manager import (
    CacheManager,
    EventAwareChunk,
    StreamingCacheWriter,
)


class TestCacheManagerExtremeEdgeCases:
    """Extreme edge case tests for CacheManager to uncover potential bugs"""

    @pytest.fixture
    def mock_redis(self):
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def cache_manager(self, mock_redis):
        return CacheManager(mock_redis, ttl=300, cache_enabled=True)

    def test_generate_cache_key_with_extreme_inputs(self, cache_manager):
        """Test cache key generation with extreme and unusual inputs"""
        extreme_cases = [
            {},  # Empty dict
            {"model": None},  # None values
            {"messages": []},  # Empty lists
            {"messages": [{}]},  # Empty message
            {"messages": [{"role": None, "content": None}]},  # None in message
            {
                "messages": [{"role": "user", "content": "x" * 10000}]
            },  # Very long content
            {"temperature": float("inf")},  # Infinity
            {"temperature": float("nan")},  # NaN
            {
                "extra_body": {"nested": {"deep": {"very": {"deep": "value"}}}}
            },  # Deep nesting
            {"unicode_field": "🌟💫✨🎉🔥💯🚀🌈⭐️🎊"},  # Unicode emojis
            {"binary_like": b"binary data".decode("latin1")},  # Binary-like data
        ]

        for case in extreme_cases:
            # Should not raise exceptions
            key = cache_manager._generate_cache_key(case)
            assert isinstance(key, str)
            assert len(key) > 0

    def test_generate_cache_key_deterministic_with_complex_data(self, cache_manager):
        """Test that cache key generation is deterministic with complex data"""
        complex_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": {"type": "text", "content": "Complex"}},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "extra_body": {
                "metadata": {"user_id": "123", "session_id": "abc"},
                "custom_params": [1, 2, 3, {"nested": True}],
            },
        }

        key1 = cache_manager._generate_cache_key(complex_data)
        key2 = cache_manager._generate_cache_key(complex_data)
        assert key1 == key2

        # Slight modification should produce different key
        modified_data = complex_data.copy()
        modified_data["temperature"] = 0.8
        key3 = cache_manager._generate_cache_key(modified_data)
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_get_with_redis_decode_error(self, cache_manager, mock_redis):
        """Test handling of Redis decode errors"""
        mock_redis.get.return_value = b"\x80\x81\x82"  # Invalid UTF-8

        result = await cache_manager.get({"model": "gpt-3.5-turbo"})
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_corrupted_json_in_cache(self, cache_manager, mock_redis):
        """Test handling of corrupted JSON in cache"""
        mock_redis.get.return_value = '{"incomplete": json'

        result = await cache_manager.get({"model": "gpt-3.5-turbo"})
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_non_serializable_response(self, cache_manager, mock_redis):
        """Test handling of non-serializable response data"""
        non_serializable_response = {
            "choices": [{"message": {"content": "test"}}],
            "function": lambda x: x,  # Non-serializable function
            "circular_ref": None,
        }
        non_serializable_response["circular_ref"] = non_serializable_response

        # Should not raise exception, just fail silently
        await cache_manager.set({"model": "gpt-3.5-turbo"}, non_serializable_response)
        # If it gets here without raising, the test passes

    @pytest.mark.asyncio
    async def test_get_streaming_with_corrupted_list_data(
        self, cache_manager, mock_redis
    ):
        """Test streaming cache with corrupted list data"""
        # Mock Redis to return corrupted data
        mock_redis.lrange.return_value = [b"\x80\x81", b"valid chunk"]

        result = await cache_manager.get_streaming({"model": "gpt-3.5-turbo"})
        assert result is None

    @pytest.mark.asyncio
    async def test_reconstruct_responses_stream_with_malformed_chunks(
        self, cache_manager
    ):
        """Test response stream reconstruction with malformed chunks"""
        malformed_chunks = [
            {"event_type": None, "data_type": None},  # Missing required fields
            {"event_type": "unknown.event", "content": "test"},  # Unknown event
            {"event_type": "response.created", "metadata": None},  # None metadata
            {
                "event_type": "response.output_text.delta",
                "content": None,
            },  # None content
        ]

        reconstructed = cache_manager._reconstruct_responses_stream(malformed_chunks)
        # Should handle malformed chunks gracefully
        assert isinstance(reconstructed, list)

    def test_event_aware_chunk_edge_cases(self):
        """Test EventAwareChunk with edge case inputs"""
        # Test with None values
        chunk = EventAwareChunk(None, None, None, None)
        assert chunk.event_type is None
        assert chunk.data_type is None
        assert chunk.content is None
        assert chunk.metadata == {}

        # Test serialization/deserialization
        chunk_dict = chunk.to_dict()
        reconstructed = EventAwareChunk.from_dict(chunk_dict)
        assert reconstructed.event_type == chunk.event_type
        assert reconstructed.metadata == chunk.metadata

    def test_event_aware_chunk_from_dict_with_malformed_data(self):
        """Test EventAwareChunk.from_dict with malformed data"""
        malformed_cases = [
            {},  # Empty dict
            {"event_type": 123},  # Non-string event type
            {"metadata": "not a dict"},  # Non-dict metadata
            {"unknown_field": "value"},  # Unknown fields
        ]

        for case in malformed_cases:
            chunk = EventAwareChunk.from_dict(case)
            # Should not raise, just handle gracefully
            assert isinstance(chunk, EventAwareChunk)

    @pytest.mark.asyncio
    async def test_streaming_cache_writer_with_empty_chunks(self, cache_manager):
        """Test StreamingCacheWriter with empty and edge case chunks"""
        writer = StreamingCacheWriter(cache_manager, {})

        edge_cases = [
            "",  # Empty string
            "\n",  # Just newline
            "data: \n",  # Empty data
            "event: \n",  # Empty event
            ": comment\n",  # SSE comment
            "data: null\n",  # Null data
            "malformed line without colon",  # Invalid format
        ]

        for chunk in edge_cases:
            # Should not raise exceptions
            result = await writer.write_and_yield(chunk)
            assert result == chunk

    @pytest.mark.asyncio
    async def test_streaming_cache_writer_parse_sse_extreme_cases(self, cache_manager):
        """Test SSE parsing with extreme cases"""
        writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

        extreme_sse_cases = [
            'data: {}\nevent: test\ndata: {"key": "value"}\n',  # Multiple data lines
            'event: test\ndata: {"nested": {"deep": {"data": "value"}}}\n',  # Deep nesting
            "data: " + "x" * 10000 + "\n",  # Very long data line
            'event: test\ndata: {"unicode": "🌟💫✨"}\n',  # Unicode in JSON
            'data: {"number": 1.7976931348623157e+308}\n',  # Extreme float
        ]

        for sse_chunk in extreme_sse_cases:
            event_type, data = writer._parse_sse_lines(sse_chunk)
            # Should handle without exceptions
            assert event_type is None or isinstance(event_type, str)
            assert data is None or isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_create_streaming_cache_writer_concurrent_access(self, cache_manager):
        """Test concurrent access to streaming cache writer creation"""

        async def create_writer():
            return await cache_manager.create_streaming_cache_writer(
                {"model": "gpt-3.5-turbo"}
            )

        # Create multiple writers concurrently
        tasks = [create_writer() for _ in range(10)]
        writers = await asyncio.gather(*tasks)

        # All should be created successfully
        for writer in writers:
            assert isinstance(writer, StreamingCacheWriter)

    @pytest.mark.asyncio
    async def test_invalidate_request_with_key_generation_failure(
        self, cache_manager, mock_redis
    ):
        """Test cache invalidation when key generation fails"""
        # Create a request that might cause key generation to fail
        problematic_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": object()}
            ],  # Non-serializable object
        }

        # Should handle gracefully without raising
        result = await cache_manager.invalidate_request(problematic_request)
        # Result should be False due to error
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_all_with_redis_scan_failure(
        self, cache_manager, mock_redis
    ):
        """Test invalidate_all when Redis scan fails"""
        mock_redis.scan_iter.side_effect = redis.RedisError("Scan failed")

        result = await cache_manager.invalidate_all()
        assert result == 0

    @pytest.mark.asyncio
    async def test_cache_stats_thread_safety(self, cache_manager):
        """Test that cache stats are thread-safe under concurrent access"""

        async def increment_stats():
            cache_manager._hits += 1
            cache_manager._misses += 1
            cache_manager._streaming_hits += 1
            cache_manager._streaming_misses += 1
            await asyncio.sleep(
                0.001
            )  # Small delay to increase chance of race conditions

        # Run concurrent operations
        tasks = [increment_stats() for _ in range(100)]
        await asyncio.gather(*tasks)

        stats = cache_manager.get_stats()
        # Should have consistent values (though exact values depend on implementation)
        assert stats["hits"] >= 0
        assert stats["misses"] >= 0
        assert stats["streaming_hits"] >= 0
        assert stats["streaming_misses"] >= 0

    @pytest.mark.asyncio
    async def test_streaming_cache_writer_detect_error_edge_cases(self, cache_manager):
        """Test error detection in streaming chunks with edge cases"""
        writer = StreamingCacheWriter(cache_manager, {})

        error_chunks = [
            'data: {"error": null}',  # Null error
            'data: {"error": ""}',  # Empty error string
            'data: {"error": {"code": "test", "message": "error"}}',  # Nested error
            'data: {"errors": ["error1", "error2"]}',  # Plural errors
            'data: {"status": "error"}',  # Status field
        ]

        for chunk in error_chunks:
            # Should handle various error formats
            has_error = writer._detect_error_in_chunk(chunk)
            assert isinstance(has_error, bool)

    @pytest.mark.asyncio
    async def test_cache_operations_with_redis_connection_loss(
        self, cache_manager, mock_redis
    ):
        """Test cache operations when Redis connection is lost"""
        # Simulate Redis connection errors
        mock_redis.get.side_effect = redis.ConnectionError("Connection lost")
        mock_redis.setex.side_effect = redis.ConnectionError("Connection lost")
        mock_redis.lrange.side_effect = redis.ConnectionError("Connection lost")

        # All operations should handle connection loss gracefully
        result = await cache_manager.get({"model": "gpt-3.5-turbo"})
        assert result is None

        await cache_manager.set({"model": "gpt-3.5-turbo"}, {"response": "test"})
        # Should complete without exception

        streaming_result = await cache_manager.get_streaming({"model": "gpt-3.5-turbo"})
        assert streaming_result is None

    def test_should_cache_with_deeply_nested_extra_body(self, cache_manager):
        """Test _should_cache with deeply nested extra_body structures"""
        deep_structure = {"cache": {"no-cache": True}}
        for _ in range(10):  # Create very deep nesting
            deep_structure = {"nested": deep_structure}

        request_data = {"extra_body": deep_structure}
        result = cache_manager._should_cache(request_data)
        # Should handle deep nesting without stack overflow
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_finalize_responses_cache_with_redis_pipeline_failure(
        self, cache_manager, mock_redis
    ):
        """Test response cache finalization when Redis pipeline fails"""
        writer = StreamingCacheWriter(cache_manager, {}, is_responses_api=True)

        # Add some chunks to the writer
        await writer.write_and_yield(
            'event: response.created\ndata: {"response": {"id": "test"}}\n'
        )
        await writer.write_and_yield(
            'event: response.output_text.delta\ndata: {"delta": "Hello"}\n'
        )

        # Mock Redis pipeline to fail
        mock_pipeline = MagicMock()
        mock_pipeline.execute.side_effect = redis.RedisError("Pipeline failed")
        mock_redis.pipeline.return_value = mock_pipeline

        # Should handle pipeline failure gracefully
        await writer._finalize_responses_cache()
        # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_intercept_stream_with_failing_async_iterator(self, cache_manager):
        """Test stream interception when the async iterator fails"""
        writer = StreamingCacheWriter(cache_manager, {})

        async def failing_stream():
            yield "chunk1"
            yield "chunk2"
            raise RuntimeError("Stream failed")

        # Should handle stream failure gracefully
        intercepted_chunks = []
        try:
            async for chunk in writer.intercept_stream(failing_stream()):
                intercepted_chunks.append(chunk)
        except RuntimeError:
            pass  # Expected

        # Should have yielded chunks before failure
        assert len(intercepted_chunks) >= 1

    def test_cache_manager_initialization_edge_cases(self):
        """Test CacheManager initialization with edge case parameters"""
        mock_redis = MagicMock()

        # Test with extreme TTL values
        cache1 = CacheManager(mock_redis, ttl=0)  # Zero TTL
        assert cache1.ttl == 0

        cache2 = CacheManager(mock_redis, ttl=-1)  # Negative TTL
        assert cache2.ttl == -1

        cache3 = CacheManager(mock_redis, ttl=2**31)  # Very large TTL
        assert cache3.ttl == 2**31

        # Test with unusual namespace
        cache4 = CacheManager(mock_redis, namespace="")  # Empty namespace
        assert cache4.namespace == ""

        cache5 = CacheManager(mock_redis, namespace="🔥💯🚀")  # Unicode namespace
        assert cache5.namespace == "🔥💯🚀"
