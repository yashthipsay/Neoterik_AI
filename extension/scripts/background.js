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
          // Try various common selectors for job descriptions
          const possibleSelectors = [
            '.job-description',
            '[data-testid="job-description"]',
            '.description',
            '#job-description',
            '[class*="job-description"]',
            '[class*="jobDescription"]',
            'section[class*="description"]',
            '[role="main"]',
            'main'
          ];
          
          for (const selector of possibleSelectors) {
            const element = document.querySelector(selector);
            if (element && element.textContent.trim().length > 100) {
              return element.textContent.trim();
            }
          }
          
          // Fallback: try to find by heading
          const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
          for (const heading of headings) {
            if (/job description|description|requirements/i.test(heading.textContent)) {
              const section = heading.nextElementSibling;
              if (section && section.textContent.trim().length > 100) {
                return section.textContent.trim();
              }
            }
          }
          
          // Last resort: return the body text with length limit
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