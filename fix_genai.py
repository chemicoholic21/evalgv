#!/usr/bin/env python3
"""
Script to diagnose and fix Google Generative AI library issues
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("Diagnosing Google Generative AI library...")
    
    # Check current installation
    print("\n1. Checking current installation...")
    success, output, error = run_command("pip show google-generativeai")
    if success:
        print("Current installation:")
        print(output)
    else:
        print("google-generativeai not found or not properly installed")
    
    # Try to import and check
    print("\n2. Testing import...")
    try:
        import google.generativeai as genai
        print("✓ Import successful")
        
        # Check available attributes
        attrs = dir(genai)
        print(f"Available attributes: {len(attrs)}")
        
        if 'configure' in attrs:
            print("✓ configure method found")
        else:
            print("✗ configure method NOT found")
            
        if 'GenerativeModel' in attrs:
            print("✓ GenerativeModel class found")
        else:
            print("✗ GenerativeModel class NOT found")
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
    
    # Recommend fix
    print("\n3. Recommended fix:")
    print("Run the following commands:")
    print("pip uninstall google-generativeai -y")
    print("pip install google-generativeai --upgrade")
    
    # Try to fix automatically
    response = input("\nWould you like to try fixing automatically? (y/n): ")
    if response.lower() == 'y':
        print("\nUninstalling current version...")
        run_command("pip uninstall google-generativeai -y")
        
        print("Installing latest version...")
        success, output, error = run_command("pip install google-generativeai --upgrade")
        
        if success:
            print("✓ Installation successful!")
            print("Please restart your Python script.")
        else:
            print("✗ Installation failed:")
            print(error)

if __name__ == "__main__":
    main()
