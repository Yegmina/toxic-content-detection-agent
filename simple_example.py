"""
Simple Example - Toxic Message Validation Agent
============================================

This example demonstrates basic usage of the production-ready toxic message validation agent.
It shows how to initialize the validator and process messages with comprehensive error handling.

Author: AI Assistant
Version: 2.0.0
"""

from message_validator import Message_Validation, ToxicValidationError, InputValidationError
import logging

def main():
    """Demonstrate basic usage of the toxic message validation agent."""
    print("🎯 Simple Toxic Message Validation Example (97.5% Accuracy)")
    print("=" * 60)
    
    try:
        # Initialize validator with production configuration
        print("Initializing validator...")
        validator = Message_Validation(
            model_path="model",
            config_path="config.json",
            enable_logging=True,
            enable_metrics=True,
            max_input_length=512
        )
        
        # Test messages with expected results
        test_cases = [
            ("COMMEND ME TY", -1, "Clean - positive gaming"),
            ("WHAT THE ACTUAL FUCK", 1, "Toxic - explicit language"),
            ("maybe you should try a different strategy", 0, "Unclear - could be advice or criticism"),
            ("fucking reported axe", 1, "Toxic - explicit + report threat"),
            ("good game everyone", -1, "Clean - sportsmanship"),
            ("you're not very good at this game", 0, "Unclear - could be honest feedback"),
            ("KYS", 1, "Toxic - kill yourself"),
            ("nice play!", -1, "Clean - positive feedback"),
            ("I hope you lose", 0, "Unclear - competitive but not explicitly toxic"),
            ("gg wp", -1, "Clean - good game well played")
        ]
        
        print("\n🔍 Testing Messages:")
        print("-" * 60)
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for message, expected_result, note in test_cases:
            try:
                # Use the new structured validation method
                result = validator.validate_message(message)
                
                # Check if prediction matches expected
                is_correct = result.result_code == expected_result
                if is_correct:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"
                
                # Display results
                print(f"{status} '{message}'")
                print(f"   Expected: {expected_result} ({['CLEAN', 'UNCLEAR', 'TOXIC'][expected_result + 1]})")
                print(f"   Got:      {result.result_code} ({result.result_text.upper()})")
                print(f"   Confidence: {result.confidence:.3f}")
                print(f"   Processing time: {result.processing_time_ms:.2f}ms")
                print(f"   Pipeline stage: {result.pipeline_stage}")
                print(f"   Note:     {note}")
                print("-" * 40)
                
            except (ToxicValidationError, InputValidationError) as e:
                print(f"❌ '{message}' - Error: {str(e)}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ '{message}' - Unexpected error: {str(e)}")
                print("-" * 40)
        
        # Display summary
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n📊 Summary:")
        print(f"   Total tests: {total_predictions}")
        print(f"   Correct: {correct_predictions}")
        print(f"   Accuracy: {accuracy:.1f}% (Target: 97.5%)")
        
        # Health check
        print(f"\n🏥 Health Check:")
        health = validator.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Initialized: {health['initialized']}")
        print(f"   Device: {health['device']}")
        
        # Performance metrics
        if validator.metrics:
            print(f"\n📈 Performance Metrics:")
            metrics = validator.get_performance_metrics()
            print(f"   Total requests: {metrics.total_requests}")
            print(f"   Successful: {metrics.successful_requests}")
            print(f"   Failed: {metrics.failed_requests}")
            print(f"   Average processing time: {metrics.average_processing_time_ms:.2f}ms")
            print(f"   Word filter hits: {metrics.word_filter_hits}")
            print(f"   Embedding hits: {metrics.embedding_hits}")
            print(f"   Fine-tuned hits: {metrics.finetuned_hits}")
            print(f"   RAG hits: {metrics.rag_hits}")
        
        print("\n✅ Example completed successfully!")
        
    except ToxicValidationError as e:
        print(f"❌ Initialization failed: {str(e)}")
        logging.error(f"Initialization error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 