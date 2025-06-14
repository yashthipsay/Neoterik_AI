'use client'

import { useEffect, useState } from 'react'
import { useSession } from 'next-auth/react'

export default function ExtensionCallback() {
  const { data: session, status } = useSession()
  const [messageSent, setMessageSent] = useState(false)

  useEffect(() => {
    if (status === 'authenticated' && session && !messageSent) {
      console.log('Extension callback: Session authenticated, sending to extension', session)
      
      // Change the page title to signal success to the background script
      document.title = "AUTH_SUCCESS";
      
      // Method 1: Direct postMessage to parent (works when opened in a popup)
      try {
        if (window.opener) {
          window.opener.postMessage(
            { 
              type: 'EXTENSION_AUTH_SUCCESS', 
              session: {
                user: session.user,
              }
            },
            '*'
          );
          console.log("Message posted to opener window");
        } else {
          console.log("No opener window found, trying broadcast");
        }
      } catch (e) {
        console.error("Failed to post message to opener:", e);
      }
      
      // Method 2: Broadcast message (works when opened in tab)
      try {
        window.postMessage(
          { 
            type: 'EXTENSION_AUTH_SUCCESS', 
            session: {
              user: session.user,
            }
          },
          '*'
        );
        console.log("Broadcast message posted");
      } catch (e) {
        console.error("Failed to broadcast message:", e);
      }
      
      // Set flag to prevent multiple messages
      setMessageSent(true);
      
      // Show success message
      document.body.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;">
          <h1 style="color:#419D78;">Successfully signed in!</h1>
          <p>You can close this window now.</p>
          <p>Signed in as: ${session.user?.name || session.user?.email}</p>
        </div>`;

        // Auto close after 3 seconds
      setTimeout(() => {
        window.close();
      }, 3000);
    }
  }, [session, status, messageSent]);

  if (status === 'loading') return <div>Loading session...</div>
  if (status === 'unauthenticated') return <div>Authentication failed</div>
  
  return <div>Authenticating...</div>
}