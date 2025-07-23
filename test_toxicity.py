#!/usr/bin/env python3
"""
Comprehensive test script for toxic message validation.
Tests various types of messages including clean, toxic, and gaming-specific content.
"""

from toxic_validation_agent import Message_Validation
import time

def test_message(validator, message: str, expected_category: str = "unknown"):
    """Test a single message and display results."""
    print(f"🔍 Testing: '{message}'")
    print(f"📋 Expected: {expected_category}")
    print("-" * 60)
    
    try:
        start_time = time.time()
        result = validator.validate_message(message)
        processing_time = (time.time() - start_time) * 1000
        
        # Display results
        print(f"📊 Results:")
        print(f"   Toxic: {'❌ YES' if result.is_toxic else '✅ NO'}")
        print(f"   Toxicity: {result.toxicity:.2%}")
        print(f"   Result Code: {result.result_code} ({result.result_text})")
        print(f"   Processing Time: {processing_time:.1f}ms")
        print(f"   Pipeline Stage: {result.pipeline_stage}")
        
        # Get detailed analysis
        print(f"\n🔍 Detailed Analysis:")
        detailed = validator.get_detailed_prediction(message)
        for stage, info in detailed.items():
            if isinstance(info, dict) and 'confidence' in info:
                print(f"   {stage}: {info['confidence']:.2%} confidence")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 Comprehensive Toxicity Detection Test")
    print("=" * 80)
    
    # Initialize validator
    print("🚀 Initializing validator...")
    start_time = time.time()
    validator = Message_Validation()
    init_time = (time.time() - start_time) * 1000
    print(f"✅ Initialized in {init_time:.1f}ms")
    print()
    
    # Test messages organized by category
    test_cases = [
        # Clean messages
        ("Hello, how are you today?", "clean"),
        ("Great game everyone, well played!", "clean"),
        ("Thanks for the help, I appreciate it!", "clean"),
        ("Nice shot! Good teamwork.", "clean"),
        ("gg wp", "clean"),
        
        # Mildly toxic messages
        ("You're not very good at this game", "mildly toxic"),
        ("Maybe you should practice more", "mildly toxic"),
        ("This team is terrible", "mildly toxic"),
        
        # Toxic messages
        ("You're such a fucking idiot!", "toxic"),
        ("Kill yourself you worthless piece of shit", "toxic"),
        ("Uninstall the game you noob", "toxic"),
        ("kys noob", "toxic"),
        ("You suck at this game", "toxic"),
        
        # Gaming-specific toxic messages
        ("uninstall noob", "gaming toxic"),
        ("kys", "gaming toxic"),
        ("git gud", "gaming toxic"),
        ("trash player", "gaming toxic"),
        ("you're garbage", "gaming toxic"),
        
        # Edge cases
        ("", "empty"),
        ("   ", "whitespace"),
        ("a" * 1000, "very long"),
        ("!@#$%^&*()", "symbols"),
        ("123456789", "numbers"),
    ]
    
    results = []
    
    for i, (message, category) in enumerate(test_cases, 1):
        print(f"📝 Test {i}/{len(test_cases)}")
        result = test_message(validator, message, category)
        results.append((message, category, result))
        print("\n" + "=" * 80 + "\n")
    
    # Summary
    print("📈 TEST SUMMARY")
    print("=" * 80)
    
    clean_count = sum(1 for _, _, r in results if r and not r.is_toxic)
    toxic_count = sum(1 for _, _, r in results if r and r.is_toxic)
    error_count = sum(1 for _, _, r in results if r is None)
    
    print(f"✅ Clean messages detected: {clean_count}")
    print(f"❌ Toxic messages detected: {toxic_count}")
    print(f"⚠️  Errors: {error_count}")
    print(f"📊 Total tests: {len(test_cases)}")
    
    # Performance metrics
    metrics = validator.get_performance_metrics()
    if metrics:
        print(f"\n⚡ Performance Metrics:")
        print(f"   Total Requests: {metrics.total_requests}")
        print(f"   Average Time: {metrics.average_processing_time_ms:.1f}ms")
        print(f"   Pipeline Hits: Word Filter={metrics.word_filter_hits}, "
              f"Embedding={metrics.embedding_hits}, "
              f"Fine-tuned={metrics.finetuned_hits}, "
              f"RAG={metrics.rag_hits}")
    
    # Detailed results by category
    print(f"\n📋 Results by Category:")
    categories = {}
    for message, category, result in results:
        if result:
            if category not in categories:
                categories[category] = []
            categories[category].append((message, result.toxicity, result.is_toxic))
    
    for category, items in categories.items():
        avg_toxicity = sum(item[1] for item in items) / len(items)
        toxic_count = sum(1 for item in items if item[2])
        print(f"   {category}: {len(items)} messages, "
              f"avg toxicity: {avg_toxicity:.2%}, "
              f"toxic: {toxic_count}/{len(items)}")

if __name__ == "__main__":
    main() 