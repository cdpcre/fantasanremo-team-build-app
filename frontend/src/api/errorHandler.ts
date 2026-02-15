import axios, { AxiosError } from 'axios'
import { logger } from '@/utils/logger'

/**
 * Error types for classification
 */
export enum ErrorType {
  NETWORK = 'NETWORK_ERROR',
  AUTH = 'AUTH_ERROR',
  VALIDATION = 'VALIDATION_ERROR',
  NOT_FOUND = 'NOT_FOUND_ERROR',
  SERVER = 'SERVER_ERROR',
  UNKNOWN = 'UNKNOWN_ERROR',
}

/**
 * Custom API error class
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public type: ErrorType,
    public statusCode?: number,
    public originalError?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Check if error is a network error (no response received)
 */
export function isNetworkError(error: unknown): error is AxiosError {
  return (
    axios.isAxiosError(error) &&
    !error.response &&
    !!error.request
  )
}

/**
 * Check if error is an authentication error (401, 403)
 */
export function isAuthError(error: unknown): boolean {
  if (axios.isAxiosError(error) && error.response) {
    return error.response.status === 401 || error.response.status === 403
  }
  return false
}

/**
 * Check if error is a validation error (400, 422)
 */
export function isValidationError(error: unknown): boolean {
  if (axios.isAxiosError(error) && error.response) {
    return error.response.status === 400 || error.response.status === 422
  }
  return false
}

/**
 * Check if error is a not found error (404)
 */
export function isNotFoundError(error: unknown): boolean {
  if (axios.isAxiosError(error) && error.response) {
    return error.response.status === 404
  }
  return false
}

/**
 * Check if error is a server error (5xx)
 */
export function isServerError(error: unknown): boolean {
  if (axios.isAxiosError(error) && error.response) {
    return error.response.status >= 500 && error.response.status < 600
  }
  return false
}

/**
 * Convert technical errors to user-friendly messages
 */
export function getUserMessage(error: unknown): string {
  // Network errors
  if (isNetworkError(error)) {
    return 'Unable to connect to the server. Please check your internet connection and try again.'
  }

  // Auth errors
  if (isAuthError(error)) {
    return 'You are not authorized to perform this action. Please log in and try again.'
  }

  // Validation errors
  if (isValidationError(error)) {
    if (axios.isAxiosError(error) && error.response?.data) {
      // Try to extract validation message from response
      const data = error.response.data as ApiErrorResponse
      if (typeof data === 'string') return data
      if (data.detail) return data.detail
      if (data.message) return data.message
      if (data.error) return data.error
      if (Array.isArray(data.non_field_errors)) {
        return data.non_field_errors.join(', ')
      }
    }
    return 'The provided data is invalid. Please check your input and try again.'
  }

  // Not found errors
  if (isNotFoundError(error)) {
    return 'The requested resource was not found. It may have been moved or deleted.'
  }

  // Server errors
  if (isServerError(error)) {
    return 'The server encountered an error. Please try again later or contact support if the problem persists.'
  }

  // Generic error
  if (error instanceof Error) {
    return error.message
  }

  return 'An unexpected error occurred. Please try again.'
}

/**
 * Type for API error response data
 */
interface ApiErrorResponse {
  detail?: string
  message?: string
  error?: string
  non_field_errors?: string[]
  [key: string]: unknown
}

/**
 * Extract error details from Axios error
 */
function getErrorDetails(error: AxiosError): {
  message: string
  details?: unknown
} {
  if (error.response?.data) {
    const data = error.response.data as ApiErrorResponse

    // Handle different response formats
    if (typeof data === 'string') {
      return { message: data }
    }

    if (data.detail) {
      return { message: data.detail, details: data }
    }

    if (data.message) {
      return { message: data.message, details: data }
    }

    if (data.error) {
      return { message: data.error, details: data }
    }

    // FastAPI validation error format
    if (data.non_field_errors) {
      return { message: data.non_field_errors.join(', '), details: data }
    }
  }

  // Default error message based on status code
  const statusMessages: Record<number, string> = {
    400: 'Bad request. Please check your input.',
    401: 'Authentication required.',
    403: 'You do not have permission to perform this action.',
    404: 'Resource not found.',
    409: 'Conflict. The resource already exists.',
    422: 'Validation error. Please check your input.',
    429: 'Too many requests. Please wait and try again.',
    500: 'Internal server error.',
    502: 'Bad gateway.',
    503: 'Service unavailable.',
    504: 'Gateway timeout.',
  }

  const message = error.response?.status
    ? statusMessages[error.response.status] || `HTTP Error ${error.response.status}`
    : error.message || 'An error occurred'

  return { message }
}

/**
 * Main error handler function for API errors
 *
 * @example
 * ```ts
 * try {
 *   await apiCall()
 * } catch (error) {
 *   handleApiError(error)
 * }
 * ```
 *
 * @example With custom callback
 * ```ts
 * try {
 *   await apiCall()
 * } catch (error) {
 *   handleApiError(error, {
 *     showDialog: true,
 *     onError: (apiError) => {
 *       // Custom error handling
 *     }
 *   })
 * }
 * ```
 */
export interface HandleApiErrorOptions {
  showDialog?: boolean
  onError?: (error: ApiError) => void
  context?: string
}

export function handleApiError(
  error: unknown,
  options: HandleApiErrorOptions = {}
): ApiError {
  const { showDialog = true, onError, context } = options

  let apiError: ApiError

  // Handle Axios errors
  if (axios.isAxiosError(error)) {
    const { message } = getErrorDetails(error)

    // Determine error type
    let errorType = ErrorType.UNKNOWN
    if (isNetworkError(error)) errorType = ErrorType.NETWORK
    else if (isAuthError(error)) errorType = ErrorType.AUTH
    else if (isValidationError(error)) errorType = ErrorType.VALIDATION
    else if (isNotFoundError(error)) errorType = ErrorType.NOT_FOUND
    else if (isServerError(error)) errorType = ErrorType.SERVER

    apiError = new ApiError(
      message,
      errorType,
      error.response?.status,
      error
    )
  }
  // Handle generic errors
  else if (error instanceof Error) {
    apiError = new ApiError(
      error.message,
      ErrorType.UNKNOWN,
      undefined,
      error
    )
  }
  // Handle unknown errors
  else {
    apiError = new ApiError(
      'An unknown error occurred',
      ErrorType.UNKNOWN,
      undefined,
      error
    )
  }

  // Log error
  const logContext = context ? `in ${context}` : ''
  logger.error(`API error ${logContext}:`, {
    type: apiError.type,
    statusCode: apiError.statusCode,
    message: apiError.message,
    originalError: apiError.originalError,
  })

  // Show error to user (in real app, you might use a toast/snackbar)
  if (showDialog) {
    const userMessage = getUserMessage(error)
    console.error('User-facing error:', userMessage)
    // In a real app, you would show a toast/notification here
    // Example: toast.error(userMessage)
  }

  // Call custom error handler
  if (onError) {
    onError(apiError)
  }

  return apiError
}

/**
 * Wrap an async function with error handling
 *
 * @example
 * ```ts
 * const safeApiCall = withErrorHandling(apiCall, {
 *   onError: (error) => {
 *     if (error.type === ErrorType.AUTH) {
 *       // Redirect to login
 *     }
 *   }
 * })
 * ```
 */
export function withErrorHandling<TArgs extends unknown[], TResult>(
  fn: (...args: TArgs) => Promise<TResult>,
  options: HandleApiErrorOptions = {}
): (...args: TArgs) => Promise<TResult> {
  return async (...args: TArgs): Promise<TResult> => {
    try {
      return await fn(...args)
    } catch (error) {
      throw handleApiError(error, options)
    }
  }
}

/**
 * Create an API interceptor for automatic error handling
 *
 * @example
 * ```ts
 * api.interceptors.response.use(
 *   (response) => response,
 *   createApiErrorInterceptor({ showDialog: true })
 * )
 * ```
 */
export function createApiErrorInterceptor(options: HandleApiErrorOptions = {}) {
  return (error: unknown) => {
    handleApiError(error, options)
    return Promise.reject(error)
  }
}

export default {
  handleApiError,
  isNetworkError,
  isAuthError,
  isValidationError,
  isNotFoundError,
  isServerError,
  getUserMessage,
  withErrorHandling,
  createApiErrorInterceptor,
  ApiError,
  ErrorType,
}
