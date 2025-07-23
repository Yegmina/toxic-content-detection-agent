#!/usr/bin/env python3
"""
Simple test script for toxic message validation.
Usage: python test_single_message.py
"""

from toxic_validation_agent import Message_Validation
import sys

def test_message(message: str):
    """Test a single message for toxicity."""
    print(f"🔍 Testing message: '{message}'")
    print("=" * 60)
    
    try:
        # Initialize the validator
        print("🚀 Initializing validator...")
        validator = Message_Validation()
        
        # Validate the message
        print("📝 Processing message...")
        result = validator.validate_message(message)
        
        # Display results
        print("\n📊 Results:")
        print(f"   Toxic: {'❌ YES' if result.is_toxic else '✅ NO'}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Result Code: {result.result_code} ({result.result_text})")
        print(f"   Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"   Pipeline Stage: {result.pipeline_stage}")
        
        # Get detailed prediction
        print("\n🔍 Detailed Analysis:")
        detailed = validator.get_detailed_prediction(message)
        for stage, info in detailed.items():
            if isinstance(info, dict) and 'confidence' in info:
                print(f"   {stage}: {info['confidence']:.2%} confidence")
        
        # Performance metrics
        metrics = validator.get_performance_metrics()
        if metrics:
            print(f"\n📈 Performance Metrics:")
            print(f"   Total Requests: {metrics.total_requests}")
            print(f"   Average Time: {metrics.average_processing_time_ms:.1f}ms")
            print(f"   Pipeline Hits: Word Filter={metrics.word_filter_hits}, "
                  f"Embedding={metrics.embedding_hits}, "
                  f"Fine-tuned={metrics.finetuned_hits}, "
                  f"RAG={metrics.rag_hits}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    """Main function to run the test."""
    if len(sys.argv) > 1:
        # Use command line argument
        message = " ".join(sys.argv[1:])
    else:
        # Use default test messages
        test_messages = [
            "Hello, how are you today?",
            "You're such a fucking idiot!",
            "Great game everyone, well played!",
            "Kill yourself you worthless piece of shit",
            "Thanks for the help, I appreciate it!",
            "Uninstall the game you noob"
        ]
        
        print("🧪 Toxicity Detection Test")
        print("=" * 60)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 Test {i}/{len(test_messages)}")
            test_message(message)
            print("\n" + "-" * 60)
        
        return
    
    # Test single message
    test_message(message)

if __name__ == "__main__":
    main() 