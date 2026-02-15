# Frontend Error Handling Implementation

## Overview

Centralized error handling system for the FantaSanremo frontend application. This implementation provides comprehensive error catching, logging, and user-friendly error messages.

## Components Created

### 1. ErrorBoundary Component
**Location:** `/frontend/src/components/ErrorBoundary.tsx`

React Error Boundary component that catches JavaScript errors anywhere in the component tree.

**Features:**
- Catches errors in component tree and displays fallback UI
- Logs errors with full context and component stack
- Provides user-friendly error message with retry/home navigation options
- Shows detailed error information in development mode
- Supports custom fallback UI
- Optional error callback for custom error handling

**Usage:**
```tsx
import ErrorBoundary from '@/components/ErrorBoundary'

// Basic usage
<ErrorBoundary>
  <App />
</ErrorBoundary>

// With custom fallback
<ErrorBoundary fallback={<CustomErrorFallback />}>
  <App />
</ErrorBoundary>

// With error callback
<ErrorBoundary
  onError={(error, errorInfo) => {
    // Send error to error tracking service
    logErrorToService(error, errorInfo)
  }}
>
  <App />
</ErrorBoundary>
```

### 2. API Error Handler
**Location:** `/frontend/src/api/errorHandler.ts`

Comprehensive error handling utilities for API calls.

**Features:**
- **Error Classification:** Classifies errors into types (Network, Auth, Validation, NotFound, Server, Unknown)
- **Helper Functions:**
  - `isNetworkError()` - Detects connectivity issues
  - `isAuthError()` - Detects 401/403 errors
  - `isValidationError()` - Detects 400/422 validation errors
  - `isNotFoundError()` - Detects 404 errors
  - `isServerError()` - Detects 5xx server errors
- **User-Friendly Messages:** `getUserMessage()` converts technical errors to user-friendly messages
- **Error Handler:** `handleApiError()` main function for processing API errors
- **Higher-Order Function:** `withErrorHandling()` wraps async functions with automatic error handling
- **Interceptor:** `createApiErrorInterceptor()` for global Axios interceptor setup

**Usage Examples:**

```tsx
import { getArtisti } from '@/api'
import {
  handleApiError,
  withErrorHandling,
  ErrorType,
  isAuthError
} from '@/api/errorHandler'

// Example 1: Basic try-catch with error handling
async function fetchData() {
  try {
    const artisti = await getArtisti()
    return artisti
  } catch (error) {
    const apiError = handleApiError(error, {
      showDialog: true,
      context: 'fetchArtisti'
    })
    // apiError contains typed error information
    throw apiError
  }
}

// Example 2: With custom error handling
try {
  await getArtisti()
} catch (error) {
  handleApiError(error, {
    showDialog: true,
    context: 'fetchArtisti',
    onError: (apiError) => {
      if (apiError.type === ErrorType.AUTH) {
        // Redirect to login
        window.location.href = '/login'
      } else if (apiError.type === ErrorType.NETWORK) {
        // Show offline UI
        showOfflineMessage()
      }
    }
  })
}

// Example 3: Wrap API function for automatic error handling
const safeGetArtisti = withErrorHandling(getArtisti, {
  showDialog: true,
  context: 'getArtisti'
})

// Now safeGetArtisti will automatically handle errors
const artisti = await safeGetArtisti()

// Example 4: Check specific error types
try {
  await getArtisti()
} catch (error) {
  if (isAuthError(error)) {
    console.log('Authentication failed')
  } else if (isNetworkError(error)) {
    console.log('Network error')
  }
}

// Example 5: Get user-friendly message
try {
  await getArtisti()
} catch (error) {
  const userMessage = getUserMessage(error)
  alert(userMessage) // "Unable to connect to the server..."
}
```

### 3. Logger Utility
**Location:** `/frontend/src/utils/logger.ts`

Simple logging utility for consistent logging across the application.

**Features:**
- Multiple log levels: DEBUG, INFO, WARN, ERROR
- Timestamp formatting
- Context/metadata support
- Development-only debug logs
- Structured logging format

**Usage:**
```tsx
import { Logger } from '@/utils/logger'

Logger.info('User logged in', { userId: 123 })
Logger.warn('API rate limit approaching', { requests: 95 })
Logger.error('Failed to load data', { endpoint: '/api/artisti' })
Logger.debug('Component state updated', { state: newState })
```

### 4. Updated Main Entry Point
**Location:** `/frontend/src/main.tsx`

Updated to wrap the application with ErrorBoundary for global error catching.

**Changes:**
```tsx
import ErrorBoundary from './components/ErrorBoundary'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
)
```

## Error Types

The error handler classifies errors into these types:

- **NETWORK_ERROR:** No response received (connectivity issues)
- **AUTH_ERROR:** 401/403 (authentication/authorization failures)
- **VALIDATION_ERROR:** 400/422 (invalid input data)
- **NOT_FOUND_ERROR:** 404 (resource not found)
- **SERVER_ERROR:** 5xx (server-side errors)
- **UNKNOWN_ERROR:** Unclassified errors

## Error Messages

User-friendly error messages are automatically generated based on error type:

| Error Type | Example Message |
|------------|-----------------|
| Network | "Unable to connect to the server. Please check your internet connection and try again." |
| Auth | "You are not authorized to perform this action. Please log in and try again." |
| Validation | "The provided data is invalid. Please check your input and try again." |
| Not Found | "The requested resource was not found. It may have been moved or deleted." |
| Server | "The server encountered an error. Please try again later or contact support if the problem persists." |

## Integration Examples

### React Component with Error Handling

```tsx
import { useState, useEffect } from 'react'
import { getArtisti } from '@/api'
import { handleApiError, ErrorType } from '@/api/errorHandler'

export function ArtistiList() {
  const [artisti, setArtisti] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadArtisti() {
      setLoading(true)
      setError(null)
      try {
        const data = await getArtisti()
        setArtisti(data)
      } catch (err) {
        const apiError = handleApiError(err, {
          showDialog: true,
          context: 'ArtistiList',
        })
        setError(apiError.message)

        // Handle specific error types
        if (apiError.type === ErrorType.NETWORK) {
          // Show offline message
        } else if (apiError.type === ErrorType.AUTH) {
          // Redirect to login
        }
      } finally {
        setLoading(false)
      }
    }

    loadArtisti()
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <ul>
      {artisti.map(artista => (
        <li key={artista.id}>{artista.nome}</li>
      ))}
    </ul>
  )
}
```

### Form Validation Error Handling

```tsx
import { validateTeam } from '@/api'
import { handleApiError, isValidationError } from '@/api/errorHandler'

export function TeamForm() {
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const result = await validateTeam(teamData)
      console.log('Team validated:', result)
    } catch (error) {
      const apiError = handleApiError(error, {
        showDialog: true,
      })

      // Handle validation errors specifically
      if (isValidationError(error) && axios.isAxiosError(error)) {
        const validationErrors = error.response?.data
        // Display field-specific errors
      }
    }
  }

  return <form onSubmit={handleSubmit}>{/* Form fields */}</form>
}
```

### Global Error Interceptor

```tsx
import { createApiErrorInterceptor } from '@/api/errorHandler'
import api from '@/api'

// Add response interceptor for global error handling
api.interceptors.response.use(
  (response) => response,
  createApiErrorInterceptor({
    showDialog: true,
    onError: (error) => {
      if (error.type === ErrorType.AUTH) {
        // Global auth handling
      }
    }
  })
)
```

## File Structure

```
frontend/src/
├── components/
│   └── ErrorBoundary.tsx          # React Error Boundary component
├── api/
│   ├── index.ts                   # Existing API functions
│   ├── errorHandler.ts            # NEW: Error handling utilities
│   └── exampleWithErrorHandling.ts # NEW: Usage examples
├── utils/
│   └── logger.ts                  # NEW: Logging utility
└── main.tsx                       # UPDATED: Wrapped with ErrorBoundary
```

## Benefits

1. **Centralized Error Handling:** All errors processed consistently
2. **User-Friendly Messages:** Technical errors converted to understandable messages
3. **Type Safety:** TypeScript support with proper error types
4. **Flexible Integration:** Multiple ways to integrate (try-catch, HOF, interceptor)
5. **Development Support:** Detailed error information in dev mode
6. **Global Error Catching:** ErrorBoundary catches component errors
7. **Logging:** Structured logging for debugging and monitoring

## Next Steps

To further enhance error handling:

1. **Add Toast/Snackbar:** Integrate a notification system for user-facing errors
2. **Error Tracking:** Add Sentry or similar service for production error tracking
3. **Retry Logic:** Add automatic retry for failed requests
4. **Offline Detection:** Add service worker for offline support
5. **Custom Fallbacks:** Create custom error fallbacks for different routes
6. **Analytics:** Track error rates and types for monitoring

## Testing

Test the error handling:

1. **Network Errors:** Disconnect internet and make API calls
2. **Auth Errors:** Make requests without authentication
3. **Validation Errors:** Submit invalid form data
4. **Not Found Errors:** Request non-existent resources
5. **Component Errors:** Throw errors in components to test ErrorBoundary

## Example Usage in Real Scenario

See `/frontend/src/api/exampleWithErrorHandling.ts` for comprehensive examples of:
- Try-catch with handleApiError
- Wrapping functions with withErrorHandling
- React component integration
- Form validation handling
- Global interceptor setup
