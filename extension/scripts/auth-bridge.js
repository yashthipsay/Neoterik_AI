console.log("Auth bridge content script loaded");

// Listen for the postMessage event from the auth callback page
window.addEventListener("message", function(event) {
  // Only accept messages from the same window
  if (event.source !== window) return;
  
  if (event.data && event.data.type === 'EXTENSION_AUTH_SUCCESS') {
    console.log("Auth bridge received authentication success:", event.data);
    
    // Forward the auth success message to the background script
    chrome.runtime.sendMessage({
      action: "authSuccess",
      session: event.data.session
      // tabId is omitted; background can use sender.tab.id if needed
    }, function(response) {
      console.log("Background script response:", response);
    });
  }
});

// Also listen for title changes
const observer = new MutationObserver(mutations => {
  if (document.title === "AUTH_SUCCESS") {
    console.log("Auth success detected via title change");
    chrome.runtime.sendMessage({
      action: "authTitleDetected"
    });
  }
});

observer.observe(document.querySelector('title'), { 
  subtree: true, 
  characterData: true, 
  childList: true 
});