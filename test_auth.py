import os
os.environ['DASHBOARD_PASSWORD'] = 'test123'
from dashboard_api import verify_password
print('Testing auth:')
print('admin:test123 ->', verify_password('admin', 'test123'))
print('admin:wrong ->', verify_password('admin', 'wrong'))
