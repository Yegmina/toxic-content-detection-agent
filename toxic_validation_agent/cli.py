"""
Command-line interface for the Toxic Content Detection Agent.

This module provides a command-line interface for easy usage
of the toxic content detection functionality.
"""

import argparse
import sys
import json
from typing import Optional

from .message_validator import Message_Validation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Toxic Content Detection Agent - Intelligent AI Agent for Real-time Content Moderation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single message
  toxic-validation "KYS"
  
  # Check multiple messages from file
  toxic-validation --file messages.txt
  
  # Get detailed output
  toxic-validation --detailed "fucking reported"
  
  # Health check
  toxic-validation --health-check
        """
    )
    
    parser.add_argument(
        "message",
        nargs="?",
        help="Message to validate for toxicity"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="File containing messages to validate (one per line)"
    )
    
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed output including confidence and pipeline stage"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Toxic Content Detection Agent 1.0.5"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize the validator
        validator = Message_Validation()
        
        # Health check
        if args.health_check:
            health = validator.health_check()
            if args.json:
                print(json.dumps(health, indent=2))
            else:
                print("🏥 Health Check Results:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
            return
        
        # Validate messages
        if args.file:
            validate_file(validator, args.file, args.detailed, args.json)
        elif args.message:
            validate_message(validator, args.message, args.detailed, args.json)
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0


def validate_message(validator: Message_Validation, message: str, detailed: bool, json_output: bool):
    """Validate a single message."""
    result = validator.validate_message(message)
    
    if json_output:
        output = {
            "message": message,
            "is_toxic": result.is_toxic,
            "confidence": result.confidence,
            "result_code": result.result_code,
            "result_text": result.result_text,
            "processing_time_ms": result.processing_time_ms,
            "pipeline_stage": result.pipeline_stage
        }
        print(json.dumps(output, indent=2))
    else:
        if detailed:
            print(f"📝 Message: {message}")
            print(f"🎯 Toxic: {result.is_toxic}")
            print(f"📊 Confidence: {result.confidence:.3f}")
            print(f"🏷️  Result: {result.result_text} ({result.result_code})")
            print(f"⏱️  Processing time: {result.processing_time_ms:.2f}ms")
            print(f"🔧 Pipeline stage: {result.pipeline_stage}")
        else:
            status = "🚫 TOXIC" if result.is_toxic else "✅ CLEAN"
            print(f"{status}: {message}")


def validate_file(validator: Message_Validation, file_path: str, detailed: bool, json_output: bool):
    """Validate messages from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}", file=sys.stderr)
        return 1
    
    results = []
    for message in messages:
        result = validator.validate_message(message)
        results.append({
            "message": message,
            "is_toxic": result.is_toxic,
            "confidence": result.confidence,
            "result_code": result.result_code,
            "result_text": result.result_text,
            "processing_time_ms": result.processing_time_ms,
            "pipeline_stage": result.pipeline_stage
        })
    
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print(f"📊 Validated {len(results)} messages:")
        print("=" * 50)
        
        toxic_count = sum(1 for r in results if r["is_toxic"])
        clean_count = len(results) - toxic_count
        
        for result in results:
            status = "🚫 TOXIC" if result["is_toxic"] else "✅ CLEAN"
            if detailed:
                print(f"{status}: {result['message']}")
                print(f"  Confidence: {result['confidence']:.3f}, Stage: {result['pipeline_stage']}")
            else:
                print(f"{status}: {result['message']}")
        
        print("=" * 50)
        print(f"📈 Summary: {clean_count} clean, {toxic_count} toxic")


if __name__ == "__main__":
    sys.exit(main()) 