'use client'

import { useEffect } from 'react'
import { useSession } from 'next-auth/react'

export default function ExtensionCallback() {
  const { data: session, status } = useSession()

  useEffect(() => {
    if (status === 'authenticated' && session) {
      console.log('Extension callback: Session authenticated, sending to extension', session)
      
      // CRITICAL: This needs to target your extension specifically
      // The * wildcard is not secure for production
      window.postMessage(
        { 
          type: 'EXTENSION_AUTH_SUCCESS', 
          session: {
            user: session.user,
          }
        },
        '*'  // In production, use your extension's URL pattern
      )
      
      // Show success message
      document.body.innerHTML = '<h1>Successfully signed in! You can close this window.</h1>'
    }
  }, [session, status])

  if (status === 'loading') return <div>Loading session...</div>
  if (status === 'unauthenticated') return <div>Authentication failed</div>
  
  return <div>Authenticating...</div>
}