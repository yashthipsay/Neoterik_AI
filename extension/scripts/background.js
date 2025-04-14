// Configuration - Update this with your actual API endpoint
const API_BASE_URL = "http://localhost:8000";

// Store currently detected job pages to avoid redundant notifications
const detectedJobPages = new Set();

// Function to check if URL is a job application page via the API
async function checkUrlWithApi(url, tabId) {
  try {
    console.log("Checking URL with API:", url);
    
    const response = await fetch(`${API_BASE_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    console.log("API Response:", data);
    
    // If it's a job application page, update the extension UI
    if (data.is_job_application) {
      // Set badge on the extension icon
      chrome.action.setBadgeText({ 
        text: "JOB", 
        tabId: tabId 
      });
      chrome.action.setBadgeBackgroundColor({ 
        color: "#419D78", 
        tabId: tabId 
      });
      
      // Store this URL as detected
      detectedJobPages.add(url);
      
      // Store job URL in local storage for the popup
      chrome.storage.local.set({ 
        currentJobPage: {
          url: url,
          detected: Date.now()
        }
      });
      
      return true;
    } else {
      // Clear badge if not a job page
      chrome.action.setBadgeText({ text: "", tabId: tabId });
      return false;
    }
  } catch (error) {
    console.error('Error checking URL with API:', error);
    return false;
  }
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkUrl') {
    const tabId = sender.tab?.id;
    if (!tabId) {
      console.error("No tab ID found in sender");
      sendResponse({ success: false, error: "No tab ID" });
      return false;
    }
    
    // Check the URL with the API
    checkUrlWithApi(request.url, tabId).then(isJobPage => {
      sendResponse({ 
        success: true, 
        isJobPage: isJobPage 
      });
    }).catch(error => {
      console.error("Error in URL check:", error);
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    });
    
    return true; // Indicates async response
  }
  
  // Handle request to extract job description
  else if (request.action === 'extractJobDescription') {
    const tabId = sender.tab?.id;
    if (!tabId) {
      sendResponse({ success: false, error: "No tab ID" });
      return false;
    }
    
    // Execute a content script to try to extract the job description
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      function: () => {
        // This function runs in the context of the page
        function findJobDescription() {
            console.log("Attempting to find 'Additional Information' or 'Cover Letter' sections...");
            
            // Look for headings or labels containing the target phrases
            const potentialLabels = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6, strong, b, label, dt, .form-label'));
            let foundText = null;
  
            for (const labelElement of potentialLabels) {
              const labelText = labelElement.textContent.trim();
              
              if (/additional information|cover letter/i.test(labelText)) {
                console.log("Found potential label:", labelText, labelElement);
                
                // Try to get text from the next sibling element
                let nextElement = labelElement.nextElementSibling;
                if (nextElement && nextElement.textContent.trim().length > 50) {
                   console.log("Found text in next sibling:", nextElement.textContent.trim());
                   foundText = nextElement.textContent.trim();
                   break; // Stop searching once found
                }
  
                // If next sibling didn't work, try the parent's next sibling (common in definition lists dt/dd)
                if (!foundText && labelElement.parentElement) {
                  nextElement = labelElement.parentElement.nextElementSibling;
                   if (nextElement && nextElement.textContent.trim().length > 50) {
                      console.log("Found text in parent's next sibling:", nextElement.textContent.trim());
                      foundText = nextElement.textContent.trim();
                      break; 
                   }
                }
                
                // If still not found, try finding a nearby textarea or content div
                // This is more complex and might need refinement based on common structures
                let parent = labelElement.parentElement;
                let attempts = 0;
                while(parent && attempts < 3) {
                   const nearbyTextarea = parent.querySelector('textarea');
                   if(nearbyTextarea && nearbyTextarea.value.trim().length > 50) {
                      console.log("Found text in nearby textarea:", nearbyTextarea.value.trim());
                      foundText = nearbyTextarea.value.trim();
                      break;
                   }
                   const nearbyDiv = parent.querySelector('div[class*="content"], div[class*="description"], div.ProseMirror'); // Common rich text editor class
                   if(nearbyDiv && nearbyDiv.textContent.trim().length > 50) {
                      console.log("Found text in nearby content div:", nearbyDiv.textContent.trim());
                      foundText = nearbyDiv.textContent.trim();
                      break;
                   }
                   parent = parent.parentElement;
                   attempts++;
                }
                if (foundText) break; 
              }
            }
  
            if (foundText) {
              return foundText;
            }
  
            console.log("Specific sections not found, falling back to body text.");
            // Fallback: return the body text if specific sections aren't found
            return document.body.innerText.substring(0, 5000);
          }
        
        return findJobDescription();
      }
    }).then(results => {
      if (results && results[0] && results[0].result) {
        sendResponse({ 
          success: true, 
          description: results[0].result 
        });
      } else {
        sendResponse({ 
          success: false, 
          error: "No job description found" 
        });
      }
    }).catch(error => {
      console.error("Error extracting job description:", error);
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    });
    
    return true; // Indicates async response
  }
});

// Check URL when tab is updated
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only check if the page has finished loading and has a valid URL
  if (changeInfo.status === 'complete' && tab.url && 
      tab.url.startsWith('http') && !tab.url.includes('chrome-extension://')) {
    checkUrlWithApi(tab.url, tabId);
  }
});

// Clear badge when tab is removed
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.action.setBadgeText({ text: "", tabId: tabId });
});