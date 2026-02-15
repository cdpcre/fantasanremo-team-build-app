# FantaSanremo Backend Tests

Comprehensive test suite for the FantaSanremo Team Builder backend API.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_api.py              # API endpoint tests
├── test_models.py           # Database model tests
└── test_ml.py               # ML features and prediction tests

pytest.ini                   # Pytest settings (in backend/)
```

## Test Coverage

### test_api.py
Tests for all REST API endpoints:

**Artist Endpoints (`/api/artisti`)**
- Get all artists with pagination and filtering
- Get single artist with historical data
- Filter by quotation range (min/max)
- Error handling for non-existent artists

**Historical Data Endpoints (`/api/storico`)**
- Get all historical editions
- Filter by year
- Get artist-specific history

**Prediction Endpoints (`/api/predizioni`)**
- Get all predictions ordered by score
- Get specific artist prediction
- Validate performer levels

**Team Endpoints (`/api/team`)**
- Validate team composition (7 artists, 100 baudi budget)
- Reject invalid teams (duplicates, budget exceeded, captain not in team)
- Simulate team scores with predictions
- Calculate captain bonus (2x multiplier)

### test_models.py
Tests for SQLAlchemy database models:

**Artista Model**
- Create artists with all fields
- Validate unique names
- Test quotation ranges
- Verify relationships (edizioni, caratteristiche, predictions)

**EdizioneFantaSanremo Model**
- Create historical editions
- Test artist-edition relationship
- Multiple editions per artist

**CaratteristicheArtista Model**
- Create artist characteristics
- One-to-one relationship validation
- Boundary values for ratings

**Predizione2026 Model**
- Create predictions
- Validate performer levels
- Confidence score ranges

**Query Operations**
- Filter by quotation ranges
- Join queries across models
- Cascade delete operations
- Table structure validation

### test_ml.py
Tests for ML feature engineering:

**Feature Engineering**
- Historical features from past performances
- Quotation-based features
- Performer level calculation (HIGH/MEDIUM/LOW)
- Data normalization
- Handle debútantes vs veterans

**Prediction Generation**
- Simplified predictions without history
- Ensemble model predictions
- Confidence calculation
- Performer level assignment

**Edge Cases**
- Empty historical data
- Single participation
- Extreme positions (1st, 30th)
- All debútantes
- Missing data handling

## ML Diagnostics (Manual)
The ML diagnostic scripts are intentionally **not** collected by pytest. Run them manually when you need end-to-end checks or data validation.

```bash
uv run python backend/ml/diagnostics/check_full_pipeline.py
uv run python backend/ml/diagnostics/check_unified_format.py
uv run python backend/ml/diagnostics/check_debuttanti_features.py
uv run python backend/ml/diagnostics/report_unified_format.py
```

## Running Tests

### Install Test Dependencies

```bash
cd backend
uv sync --extra dev
```

### Run All Tests

```bash
cd backend
pytest
```

### Run Specific Test File

```bash
pytest tests/test_api.py
```

### Run Specific Test Class

```bash
pytest tests/test_api.py::TestArtistEndpoint
```

### Run Specific Test Function

```bash
pytest tests/test_api.py::TestArtistEndpoint::test_get_all_artisti_success
```

### Run with Coverage Report

```bash
pytest --cov=. --cov-report=html --cov-report=term
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only API tests
pytest -m api

# Run only model tests
pytest -m models

# Run only ML tests
pytest -m ml

# Run only edge case tests
pytest -m edge
```

### Run Tests in Parallel (Faster)

```bash
pytest -n auto
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Detailed Error Traces

```bash
pytest --tb=long
```

## Fixtures

### Database Fixtures
- `db_session`: In-memory SQLite database for each test
- `client`: FastAPI test client with mocked database

### Data Fixtures
- `sample_artisti`: 10 sample artists with various quotations
- `sample_storico`: Historical editions for sample artists
- `sample_caratteristiche`: Artist characteristics
- `sample_predizioni`: 2026 predictions for all artists
- `valid_team_data`: Valid team configuration
- `budget_exceeded_team_data`: Team exceeding 100 baudi
- `simulate_team_data`: 5-artist team for simulation

## Test Categories

### Positive Tests
- Valid inputs and expected behavior
- Successful API responses
- Correct data storage and retrieval

### Negative Tests
- Invalid inputs
- Missing required fields
- Constraint violations
- Error handling

### Edge Cases
- Boundary values (min/max)
- Empty datasets
- Single records
- Extreme values

### Integration Tests
- Complete workflows
- Multi-step operations
- Error recovery

## Key Test Cases

### API Tests
- GET `/api/artisti` returns filtered list
- GET `/api/artisti/{id}` returns artist with history
- POST `/api/team/validate` validates 100 baudi budget
- POST `/api/team/validate` rejects teams >100 baudi
- POST `/api/team/validate` rejects duplicate artists
- POST `/api/team/simulate` calculates correct scores

### Model Tests
- Artist names are unique
- Relationships work bidirectionally
- Cascade deletes work correctly
- Queries filter and sort properly

### ML Tests
- Feature engineering handles veterans and debútantes
- Performer levels match historical performance
- Predictions generate valid scores
- Normalization handles missing values

## Debugging Failed Tests

### Run with Python Debugger
```bash
pytest --pdb
```

### Stop at First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

### Print Debug Output
```bash
pytest -s
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    cd backend
    uv sync --extra dev
    uv run pytest --cov=. --cov-report=xml
```

If you want ML diagnostics in CI, call the scripts explicitly (they are not part of pytest collection):

```bash
uv run python backend/ml/diagnostics/report_unified_format.py
```

## Test Statistics

- **Total Tests**: 100+
- **Test Files**: 3
- **Coverage Target**: >80%

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use fixtures for common test data
3. **Clear Names**: Test names should describe what they test
4. **AAA Pattern**: Arrange, Act, Assert
5. **Mock External**: Dependencies where appropriate
6. **Test Boundaries**: Edge cases and invalid inputs

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Add fixtures for new test data
4. Update this README
5. Maintain >80% coverage

## Troubleshooting

**Import Errors**: Ensure you're running from the `backend/` directory

**Database Errors**: Tests use in-memory SQLite, no external DB needed

**ML Model Errors**: ML tests mock models, no trained models required

**Slow Tests**: Use `-m unit` for fast unit tests only
