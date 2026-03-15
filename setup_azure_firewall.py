"""
Dynamically manage Azure SQL Server firewall rules for GitHub Actions runners.
Adds current runner's IP to allow database connections.
"""

import os
import requests
import json
from datetime import datetime
from azure.identity import ClientSecretCredential

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
    """Create an Azure SQL Server firewall rule using REST API."""
    try:
        # Get Azure credentials from environment
        tenant_id = os.getenv("AZURE_SQL_TENANT_ID")
        client_id = os.getenv("AZURE_SQL_CLIENT_ID")
        client_secret = os.getenv("AZURE_SQL_CLIENT_SECRET")
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "default-rg")
        server_name = "cryptopp"
        
        if not all([tenant_id, client_id, client_secret, subscription_id]):
            raise ValueError("Missing Azure credentials in environment variables (need TENANT_ID, CLIENT_ID, CLIENT_SECRET, SUBSCRIPTION_ID)")
        
        # Get access token using ClientSecretCredential
        print("Authenticating with Azure...")
        credentials = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Get token for Azure Management API
        token = credentials.get_token("https://management.azure.com/.default")
        access_token = token.token
        print("✓ Azure authentication successful")
        
        # Build REST API URL for firewall rule
        api_url = (
            f"https://management.azure.com/subscriptions/{subscription_id}/"
            f"resourceGroups/{resource_group}/"
            f"providers/Microsoft.Sql/servers/{server_name}/"
            f"firewallRules/{rule_name}"
            f"?api-version=2015-05-01-preview"
        )
        
        # Prepare firewall rule request body
        firewall_rule_body = {
            "properties": {
                "startIpAddress": ip_address,
                "endIpAddress": ip_address
            }
        }
        
        # Create or update firewall rule via REST API
        print(f"Creating firewall rule '{rule_name}' for IP {ip_address}...")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.put(
            api_url,
            json=firewall_rule_body,
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            print(f"✓ Firewall rule created/updated successfully")
            return True
        elif response.status_code == 409:
            # Conflict - rule already exists, try to update anyway
            print(f"✓ Firewall rule already exists, updating...")
            response = requests.put(
                api_url,
                json=firewall_rule_body,
                headers=headers,
                timeout=30
            )
            if response.status_code in [200, 201]:
                print(f"✓ Firewall rule updated successfully")
                return True
            else:
                error_msg = response.text
                print(f"✗ Failed to update firewall rule: {error_msg}")
                raise Exception(f"Firewall rule update failed: {error_msg}")
        else:
            error_msg = response.text
            print(f"✗ Failed to create firewall rule: {error_msg}")
            raise Exception(f"Firewall rule creation failed: {error_msg}")
    
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
