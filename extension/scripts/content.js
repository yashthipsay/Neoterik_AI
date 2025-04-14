console.log("Neoterik Cover Letter Assistant: Content script loaded");

// Check URL with background script
function checkCurrentUrl() {
  const url = window.location.href;
  
  chrome.runtime.sendMessage({
    action: 'checkUrl',
    url: url
  }, response => {
    if (chrome.runtime.lastError) {
      console.error("Error sending message:", chrome.runtime.lastError);
      return;
    }
    
    if (response && response.success && response.isJobPage) {
      console.log("Job page detected!");
      
      // Inject a notification if not already present
      if (!document.getElementById('neoterik-job-detected')) {
        injectJobPageNotification();
      }
    }
  });
}

// Inject notification banner
function injectJobPageNotification() {
  // Create notification element
  const notification = document.createElement('div');
  notification.id = 'neoterik-job-detected';
  notification.style.cssText = `
    position: fixed;
    top: 16px;
    right: 16px;
    background-color: #419D78;
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    font-size: 14px;
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    max-width: 300px;
  `;
  
  // Create the content
  notification.innerHTML = `
    <div style="margin-right: 12px;">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M21 7V17C21 18.1046 20.1046 19 19 19H5C3.89543 19 3 18.1046 3 17V7M21 7C21 5.89543 20.1046 5 19 5H5C3.89543 5 3 5.89543 3 7M21 7L12 13L3 7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div>
      <div style="font-weight: 600; margin-bottom: 4px;"></div>
      <div style="font-size: 12px;">Click to generate a cover letter</div>
    </div>
  `;
  
  // Add click handler
  notification.addEventListener('click', () => {
    // Extract job description from the page
    chrome.runtime.sendMessage({
      action: 'extractJobDescription'
    }, response => {
      if (response && response.success) {
        // Store the description for the popup to use
        chrome.storage.local.set({
          jobDescription: response.description
        }, () => {
          // Open the extension popup
          chrome.runtime.sendMessage({ action: 'openPopup' });
        });
      }
    });
    
    // Remove notification after click
    notification.remove();
  });
  
  // Add dismiss button
  const dismissBtn = document.createElement('button');
  dismissBtn.style.cssText = `
    background: transparent;
    border: none;
    color: white;
    font-size: 16px;
    cursor: pointer;
    margin-left: 12px;
    padding: 0 4px;
  `;
  dismissBtn.innerHTML = '×';
  dismissBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    notification.remove();
  });
  notification.appendChild(dismissBtn);
  
  // Add to page
  document.body.appendChild(notification);
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.remove();
    }
  }, 10000);
}

// Run on page load
checkCurrentUrl();

// Also listen for SPA navigation
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    checkCurrentUrl();
  }
}).observe(document, {subtree: true, childList: true});

// Listen for popstate events (history navigation)
window.addEventListener('popstate', checkCurrentUrl);