/**
 * Example demonstrating how to use the centralized error handling
 *
 * This file shows how to integrate errorHandler.ts with existing API calls
 */

import { getArtisti } from './index'
import { handleApiError, withErrorHandling, ErrorType } from './errorHandler'

// Example 1: Using try-catch with handleApiError
export async function fetchArtistiWithErrorHandling(params?: { min_quotazione?: number; max_quotazione?: number }) {
  try {
    const artisti = await getArtisti(params)
    return artisti
  } catch (error) {
    const apiError = handleApiError(error, {
      showDialog: true,
      context: 'fetchArtisti',
      onError: (error) => {
        // Custom error handling logic
        if (error.type === ErrorType.NETWORK) {
          console.log('Network error detected, showing offline UI')
        } else if (error.type === ErrorType.AUTH) {
          console.log('Auth error detected, redirecting to login')
          // window.location.href = '/login'
        }
      }
    })
    throw apiError
  }
}

// Example 2: Wrapping API function with withErrorHandling
export const safeGetArtisti = withErrorHandling(getArtisti, {
  showDialog: true,
  context: 'getArtisti',
  onError: (error) => {
    if (error.type === ErrorType.VALIDATION) {
      console.log('Validation error:', error.message)
    }
  }
})

// Example 3: Using in a React component
/*
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
*/

// Example 4: Form submission with validation error handling
// NOTE: JSX example code moved to separate .md.example file to avoid TypeScript compilation issues
// See exampleWithErrorHandling.example.md for the full React component example.

// Example 5: Setting up global error interceptor
/*
import { createApiErrorInterceptor } from '@/api/errorHandler'
import api from '@/api'

// Add response interceptor to handle all API errors globally
api.interceptors.response.use(
  (response) => response,
  createApiErrorInterceptor({
    showDialog: true,
    onError: (error) => {
      // Global error handling
      if (error.type === ErrorType.AUTH) {
        // Redirect to login or show auth modal
      }
    }
  })
)
*/

export default {
  fetchArtistiWithErrorHandling,
  safeGetArtisti,
}
