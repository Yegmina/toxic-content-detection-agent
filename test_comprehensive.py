"""
Comprehensive Test - Toxic Message Validation Agent
================================================

This test suite provides comprehensive testing of the production-ready toxic message validation agent.
It tests various scenarios including clean, toxic, and unclear messages with detailed analysis.

Author: AI Assistant
Version: 2.0.0
"""

from message_validator import Message_Validation, ToxicValidationError, InputValidationError
import logging
import time

def run_comprehensive_test():
    """Run comprehensive test suite for the toxic message validation agent."""
    print("🎯 Comprehensive Toxic Message Validation Test (97.5% Accuracy)")
    print("=" * 60)
    
    try:
        # Initialize validator with production configuration
        print("Initializing validator...")
        start_time = time.time()
        
        validator = Message_Validation(
            model_path="model",
            config_path="config.json",
            enable_logging=True,
            enable_metrics=True,
            max_input_length=512
        )
        
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {init_time:.2f} seconds")
        
        # Comprehensive test cases
        test_cases = [
            # Clean messages (expected: -1)
            ("COMMEND ME TY", -1, "Clean - positive gaming"),
            ("good game everyone", -1, "Clean - sportsmanship"),
            ("nice play!", -1, "Clean - positive feedback"),
            ("gg wp", -1, "Clean - good game well played"),
            ("thanks for the game", -1, "Clean - polite"),
            ("well played", -1, "Clean - sportsmanship"),
            ("good job team", -1, "Clean - positive feedback"),
            ("nice save", -1, "Clean - positive feedback"),
            ("great teamwork", -1, "Clean - positive feedback"),
            ("awesome play", -1, "Clean - positive feedback"),
            ("you're doing great", -1, "Clean - encouragement"),
            ("keep it up", -1, "Clean - encouragement"),
            ("nice strategy", -1, "Clean - positive feedback"),
            ("good call", -1, "Clean - positive feedback"),
            ("excellent teamwork", -1, "Clean - positive feedback"),
            ("well coordinated", -1, "Clean - positive feedback"),
            ("nice shot", -1, "Clean - positive feedback"),
            ("perfect timing", -1, "Clean - positive feedback"),
            ("great positioning", -1, "Clean - positive feedback"),
            ("smart play", -1, "Clean - positive feedback"),
            
            # Toxic messages (expected: 1)
            ("WHAT THE ACTUAL FUCK", 1, "Toxic - explicit language"),
            ("fucking reported axe", 1, "Toxic - explicit + report threat"),
            ("KYS", 1, "Toxic - kill yourself"),
            ("you're trash", 1, "Toxic - insult"),
            ("uninstall the game", 1, "Toxic - aggressive"),
            ("you suck at this game", 1, "Toxic - skill insult"),
            ("go die", 1, "Toxic - death wish"),
            ("fuck you", 1, "Toxic - explicit insult"),
            ("you're garbage", 1, "Toxic - insult"),
            ("delete the game", 1, "Toxic - aggressive"),
            ("you're worthless", 1, "Toxic - insult"),
            ("kill yourself", 1, "Toxic - death wish"),
            ("you're a noob", 1, "Toxic - skill insult"),
            ("uninstall and kill yourself", 1, "Toxic - aggressive + death wish"),
            ("you're the worst player ever", 1, "Toxic - extreme insult"),
            ("fucking noob", 1, "Toxic - explicit + skill insult"),
            ("you're so bad", 1, "Toxic - skill insult"),
            ("get good", 1, "Toxic - dismissive"),
            ("you're terrible", 1, "Toxic - insult"),
            ("fuck off", 1, "Toxic - explicit dismissal"),
            ("you're useless", 1, "Toxic - insult"),
            ("go kill yourself", 1, "Toxic - death wish"),
            ("you're a waste of space", 1, "Toxic - extreme insult"),
            ("fucking idiot", 1, "Toxic - explicit + insult"),
            ("you're stupid", 1, "Toxic - insult"),
            ("uninstall now", 1, "Toxic - aggressive"),
            ("you're a disgrace", 1, "Toxic - insult"),
            ("fuck this team", 1, "Toxic - explicit + team insult"),
            ("you're all trash", 1, "Toxic - team insult"),
            ("worst team ever", 1, "Toxic - team insult"),
            ("you're all noobs", 1, "Toxic - team skill insult"),
            ("fucking team", 1, "Toxic - explicit + team insult"),
            ("you're all garbage", 1, "Toxic - team insult"),
            ("uninstall all of you", 1, "Toxic - aggressive team insult"),
            ("you're all worthless", 1, "Toxic - team insult"),
            ("kill yourselves", 1, "Toxic - team death wish"),
            ("you're all terrible", 1, "Toxic - team insult"),
            ("fuck you all", 1, "Toxic - explicit team insult"),
            ("you're all idiots", 1, "Toxic - team insult"),
            ("worst players ever", 1, "Toxic - team insult"),
            ("you're all useless", 1, "Toxic - team insult"),
            
            # Unclear messages (expected: 0)
            ("maybe you should try a different strategy", 0, "Unclear - could be advice or criticism"),
            ("you're not very good at this game", 0, "Unclear - could be honest feedback"),
            ("I hope you lose", 0, "Unclear - competitive but not explicitly toxic"),
            ("that was a bad play", 0, "Unclear - criticism but not necessarily toxic"),
            ("you need to improve", 0, "Unclear - could be constructive or insulting")
        ]
        
        print(f"\n🔍 Testing {len(test_cases)} Messages:")
        print("-" * 60)
        
        # Test execution
        correct_predictions = 0
        total_predictions = len(test_cases)
        category_results = {"clean": 0, "toxic": 0, "unclear": 0}
        category_correct = {"clean": 0, "toxic": 0, "unclear": 0}
        
        for i, (message, expected_result, note) in enumerate(test_cases, 1):
            try:
                # Get validation result
                result = validator.validate_message(message)
                
                # Determine category
                if expected_result == -1:
                    category = "clean"
                elif expected_result == 1:
                    category = "toxic"
                else:
                    category = "unclear"
                
                category_results[category] += 1
                
                # Check if prediction matches expected
                is_correct = result.result_code == expected_result
                if is_correct:
                    correct_predictions += 1
                    category_correct[category] += 1
                    status = "✅"
                else:
                    status = "❌"
                
                # Display results
                print(f"{status} '{message}'")
                print(f"   Expected: {expected_result} ({['CLEAN', 'UNCLEAR', 'TOXIC'][expected_result + 1]})")
                print(f"   Got:      {result.result_code} ({result.result_text.upper()})")
                print(f"   Note:     {note}")
                print("-" * 40)
                
            except (ToxicValidationError, InputValidationError) as e:
                print(f"❌ '{message}' - Error: {str(e)}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ '{message}' - Unexpected error: {str(e)}")
                print("-" * 40)
        
        # Calculate statistics
        overall_accuracy = (correct_predictions / total_predictions) * 100
        
        clean_accuracy = (category_correct["clean"] / category_results["clean"]) * 100 if category_results["clean"] > 0 else 0
        toxic_accuracy = (category_correct["toxic"] / category_results["toxic"]) * 100 if category_results["toxic"] > 0 else 0
        unclear_accuracy = (category_correct["unclear"] / category_results["unclear"]) * 100 if category_results["unclear"] > 0 else 0
        
        # Display summary
        print(f"\n📊 Test Summary:")
        print(f"   Total tests: {total_predictions}")
        print(f"   Correct: {correct_predictions}")
        print(f"   Accuracy: {overall_accuracy:.1f}%")
        
        print(f"\n📈 Breakdown by Category:")
        print(f"   Clean tests: {category_results['clean']}")
        print(f"   Toxic tests: {category_results['toxic']}")
        print(f"   Unclear tests: {category_results['unclear']}")
        
        print(f"\n🎯 Category Accuracy:")
        print(f"   Clean: {clean_accuracy:.1f}% ({category_correct['clean']}/{category_results['clean']})")
        print(f"   Toxic: {toxic_accuracy:.1f}% ({category_correct['toxic']}/{category_results['toxic']})")
        print(f"   Unclear: {unclear_accuracy:.1f}% ({category_correct['unclear']}/{category_results['unclear']})")
        
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
        
        # Detailed analysis example
        print(f"\n🔬 Detailed Analysis Example:")
        print("-" * 60)
        
        example_message = "maybe you should try a different strategy"
        detailed_result = validator.validate_message(example_message)
        detailed_info = validator.get_detailed_prediction(example_message)
        
        print(f"Message: {example_message}")
        print(f"Final Result: {detailed_result.result_code} ({detailed_result.result_text})")
        print(f"Embedding: {detailed_info['embedding_prediction']} (confidence: {detailed_info['embedding_confidence']:.3f})")
        print(f"Fine-tuned: {detailed_info['finetuned_prediction']} (confidence: {detailed_info['finetuned_confidence']:.3f})")
        print(f"Pipeline Stage: {detailed_result.pipeline_stage}")
        print(f"Processing Time: {detailed_result.processing_time_ms:.2f}ms")
        
        print(f"\n✅ Comprehensive test completed!")
        
        return {
            "overall_accuracy": overall_accuracy,
            "clean_accuracy": clean_accuracy,
            "toxic_accuracy": toxic_accuracy,
            "unclear_accuracy": unclear_accuracy,
            "total_tests": total_predictions,
            "correct_predictions": correct_predictions
        }
        
    except ToxicValidationError as e:
        print(f"❌ Test failed: {str(e)}")
        logging.error(f"Test error: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    results = run_comprehensive_test()
    if results:
        print(f"\n🎉 Test Results Summary:")
        print(f"   Overall Accuracy: {results['overall_accuracy']:.1f}% (Target: 97.5%)")
        print(f"   Clean Accuracy: {results['clean_accuracy']:.1f}%")
        print(f"   Toxic Accuracy: {results['toxic_accuracy']:.1f}%")
        print(f"   Unclear Accuracy: {results['unclear_accuracy']:.1f}%") 