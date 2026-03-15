"""
Dynamically manage Azure SQL Server firewall rules for GitHub Actions runners.
Adds current runner's IP to allow database connections.
"""

import os
import requests
import subprocess
import json
from datetime import datetime

def get_runner_ip():
    """Get the current GitHub Actions runner's public IP."""
    try:
        # Check multiple IP providers for reliability
        providers = [
            "https://api.ipify.org?format=json",
            "https://httpbin.org/ip",
            "https://icanhazip.com"
        ]
        
        for provider in providers:
            try:
                if "icanhazip" in provider:
                    response = requests.get(provider, timeout=5)
                    ip = response.text.strip()
                else:
                    response = requests.get(provider, timeout=5)
                    data = response.json()
                    ip = data.get("origin") or data.get("ip")
                
                if ip:
                    print(f"✓ Runner IP detected: {ip}")
                    return ip
            except Exception as e:
                print(f"  Failed to get IP from {provider}: {e}")
                continue
        
        raise Exception("Could not determine runner IP from any provider")
    except Exception as e:
        print(f"✗ Error getting runner IP: {e}")
        raise

def create_firewall_rule(ip_address, rule_name="GitHubActions"):
    """Create an Azure SQL Server firewall rule for the given IP address."""
    try:
        # Get Azure credentials from environment
        tenant_id = os.getenv("AZURE_SQL_TENANT_ID")
        client_id = os.getenv("AZURE_SQL_CLIENT_ID")
        client_secret = os.getenv("AZURE_SQL_CLIENT_SECRET")
        
        if not all([tenant_id, client_id, client_secret]):
            raise ValueError("Missing Azure credentials in environment variables")
        
        # Get Azure access token
        print("Authenticating with Azure...")
        token_response = subprocess.run(
            [
                "az", "login",
                "--service-principal",
                "-u", client_id,
                "-p", client_secret,
                "--tenant", tenant_id
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if token_response.returncode != 0:
            print(f"✗ Azure login failed: {token_response.stderr}")
            raise Exception(f"Azure authentication failed: {token_response.stderr}")
        
        print("✓ Azure authentication successful")
        
        # Set default subscription
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        if subscription_id:
            print(f"Setting default subscription to {subscription_id}...")
            set_sub_response = subprocess.run(
                ["az", "account", "set", "--subscription", subscription_id],
                capture_output=True,
                text=True,
                check=False
            )
            if set_sub_response.returncode != 0:
                print(f"✗ Failed to set subscription: {set_sub_response.stderr}")
                raise Exception(f"Failed to set subscription: {set_sub_response.stderr}")
            print("✓ Subscription set successfully")
        
        # Extract server name from connection string
        server_name = "cryptopp"
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "default-rg")
        
        # Create or update firewall rule
        print(f"Creating firewall rule '{rule_name}' for IP {ip_address}...")
        
        rule_response = subprocess.run(
            [
                "az", "sql", "server", "firewall-rule", "create",
                "--resource-group", resource_group,
                "--server", server_name,
                "--name", rule_name,
                "--start-ip-address", ip_address,
                "--end-ip-address", ip_address
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        if rule_response.returncode == 0:
            print(f"✓ Firewall rule created/updated successfully")
            return True
        elif "already exists" in rule_response.stderr or "duplicate" in rule_response.stderr.lower():
            print(f"✓ Firewall rule already exists (this is fine)")
            # Try to update it instead
            update_response = subprocess.run(
                [
                    "az", "sql", "server", "firewall-rule", "update",
                    "--resource-group", resource_group,
                    "--server", server_name,
                    "--name", rule_name,
                    "--start-ip-address", ip_address,
                    "--end-ip-address", ip_address
                ],
                capture_output=True,
                text=True,
                check=False
            )
            if update_response.returncode == 0:
                print(f"✓ Firewall rule updated successfully")
                return True
            else:
                print(f"✗ Failed to update firewall rule: {update_response.stderr}")
                raise Exception(f"Firewall rule update failed: {update_response.stderr}")
        else:
            print(f"✗ Failed to create firewall rule: {rule_response.stderr}")
            raise Exception(f"Firewall rule creation failed: {rule_response.stderr}")
    
    except Exception as e:
        print(f"✗ Error creating firewall rule: {e}")
        raise

def main():
    """Main function to setup Azure firewall for GitHub Actions."""
    print(f"[{datetime.now().isoformat()}] Setting up Azure SQL firewall for GitHub Actions runner")
    print("-" * 70)
    
    try:
        # Step 1: Get runner IP
        runner_ip = get_runner_ip()
        
        # Step 2: Create firewall rule
        create_firewall_rule(runner_ip)
        
        print("-" * 70)
        print(f"[{datetime.now().isoformat()}] ✓ Firewall setup completed successfully")
        print(f"   Runner IP {runner_ip} is now allowed to access the database")
        return 0
    
    except Exception as e:
        print("-" * 70)
        print(f"[{datetime.now().isoformat()}] ✗ Firewall setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
