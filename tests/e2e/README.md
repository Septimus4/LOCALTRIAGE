# LOCALTRIAGE E2E Tests

End-to-end API tests using Postman/Newman for the LOCALTRIAGE Customer Support Triage platform.

## Overview

This test suite validates the complete API functionality including:

- **Health & Status** - API health checks
- **Ticket Triage** - Classification and priority assignment
- **Draft Generation** - Response drafting with LLM and templates
- **Similar Tickets** - Semantic search functionality
- **Feedback** - User feedback collection
- **Metrics & Analytics** - System metrics endpoints
- **Tickets** - Ticket listing and retrieval
- **Error Handling** - Validation and error responses
- **Performance** - Response time validation

## Files

| File | Description |
|------|-------------|
| `postman_collection.json` | Main Postman collection with all test cases |
| `postman_environment_local.json` | Environment for local development |
| `postman_environment_staging.json` | Environment for staging |
| `run_e2e_tests.sh` | Shell script to run tests via Newman |

## Prerequisites

### Option 1: Postman Desktop App
1. Import `postman_collection.json` into Postman
2. Import the desired environment file
3. Run the collection manually

### Option 2: Newman CLI
```bash
# Install Newman globally
npm install -g newman newman-reporter-htmlextra

# Or install locally
npm install newman newman-reporter-htmlextra
```

## Running Tests

### Using the Shell Script

```bash
# Run against local environment (localhost:8000)
./run_e2e_tests.sh

# Run against staging
./run_e2e_tests.sh -e staging

# Run against custom URL
./run_e2e_tests.sh -u http://localhost:9000

# Run specific test folder
./run_e2e_tests.sh -f "Health & Status"

# Stop on first failure
./run_e2e_tests.sh -b

# Show help
./run_e2e_tests.sh -h
```

### Using Newman Directly

```bash
# Basic run
newman run postman_collection.json \
  --environment postman_environment_local.json

# With HTML report
newman run postman_collection.json \
  --environment postman_environment_local.json \
  --reporters cli,htmlextra \
  --reporter-htmlextra-export ./report.html

# With JUnit XML (for CI)
newman run postman_collection.json \
  --environment postman_environment_local.json \
  --reporters cli,junit \
  --reporter-junit-export ./results.xml

# Override base URL
newman run postman_collection.json \
  --environment postman_environment_local.json \
  --env-var "base_url=http://custom-url:8000"
```

## Test Scenarios

### 1. Health Check
- Verifies API is responding
- Checks health status fields
- Validates response time

### 2. Triage Endpoints
- Technical issue classification
- Billing issue classification  
- Account issue classification
- Validation error handling (missing fields, empty body)

### 3. Draft Generation
- LLM-powered draft generation
- Template-only draft generation
- Draft with category hints
- Response structure validation

### 4. Similar Tickets
- Semantic search functionality
- Result structure validation

### 5. Feedback Collection
- Positive feedback submission
- Negative feedback with corrections
- Feedback ID generation

### 6. Metrics
- Daily metrics retrieval
- Weekly metrics retrieval
- Monthly metrics retrieval
- Invalid period handling

### 7. Ticket Management
- List tickets with pagination
- Filter by category
- Filter by priority
- Get single ticket by ID

### 8. Error Handling
- Invalid JSON handling
- Invalid parameters
- Non-existent endpoints

### 9. Performance
- Triage response time validation
- Draft generation time validation

## Test Variables

The collection uses the following variables:

| Variable | Description |
|----------|-------------|
| `base_url` | API base URL |
| `ticket_id` | Stored from triage response for subsequent tests |
| `draft_id` | Stored from draft response for feedback tests |

## Reports

Tests generate the following reports:

- **HTML Report**: Interactive report with detailed results (`reports/e2e/e2e-report.html`)
- **JUnit XML**: CI-friendly format (`reports/e2e/e2e-results.xml`)
- **CLI Output**: Real-time test progress

## CI/CD Integration

The E2E tests are integrated into GitHub Actions:

- **Trigger**: Push to main/master branch, changes to `src/api/**` or `tests/e2e/**`
- **Manual**: Can be triggered manually via workflow_dispatch
- **Artifacts**: HTML and XML reports are uploaded as artifacts

See `.github/workflows/e2e-tests.yml` for the complete workflow.

## Adding New Tests

1. Open the collection in Postman
2. Add new request to appropriate folder
3. Add test scripts in the "Tests" tab
4. Export and save the updated collection

### Test Script Template

```javascript
pm.test('Status code is 200', function () {
    pm.response.to.have.status(200);
});

pm.test('Response has required field', function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('field_name');
});

pm.test('Response time is acceptable', function () {
    pm.expect(pm.response.responseTime).to.be.below(2000);
});
```

## Troubleshooting

### API Not Responding
```bash
# Check if API is running
curl http://localhost:8000/health

# Start the API
cd /path/to/project
PYTHONPATH=src python -m uvicorn api.api:app --reload
```

### Newman Not Found
```bash
npm install -g newman newman-reporter-htmlextra
```

### Permission Denied on Script
```bash
chmod +x run_e2e_tests.sh
```
