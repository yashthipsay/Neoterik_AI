// No changes needed here based on the previous version, but ensure it matches:
document.addEventListener("DOMContentLoaded", function () {
  console.log("Popup DOM loaded.");
  const signinButton = document.getElementById("signin-btn");
  const signoutButton = document.getElementById("signout-btn");
  const userInfoDiv = document.getElementById("user-info");
  const mainContentDiv = document.getElementById("main-content");
  const authRequiredMessageDiv = document.getElementById(
    "auth-required-message"
  );

  // Function to update UI based on login state
  function updateUI(isLoggedIn, user) {
    console.log("Updating UI. isLoggedIn:", isLoggedIn, "User:", user);

    // Get fresh references to DOM elements
    const signinButton = document.getElementById("signin-btn");
    const signoutButton = document.getElementById("signout-btn");
    const userInfoDiv = document.getElementById("user-info");
    const mainContentDiv = document.getElementById("main-content");
    const authRequiredMessageDiv = document.getElementById(
      "auth-required-message"
    );

    if (isLoggedIn && user) {
      // Signed in state
      if (signinButton) signinButton.style.display = "none";
      if (signoutButton) signoutButton.style.display = "block";
      if (userInfoDiv) {
        userInfoDiv.style.display = "flex"; // Use flex for aligning user info and avatar

        const displayName = user?.name?.split(" ")[0] || user?.email || "User";

        // Create HTML content with avatar and name
        let userContent = "";

        // Add avatar if available
        if (user.image) {
          userContent += `<img src="${user.image}" alt="Profile" class="user-avatar">`;
        }

        // Add display name
        userContent += `Hi, ${displayName}!`;

        // Set the HTML content
        userInfoDiv.innerHTML = userContent;
      }
      if (mainContentDiv) mainContentDiv.style.display = "block";
      if (authRequiredMessageDiv) authRequiredMessageDiv.style.display = "none";
    } else {
      // Signed out state
      if (signinButton) signinButton.style.display = "block";
      if (signoutButton) signoutButton.style.display = "none";
      if (userInfoDiv) {
        userInfoDiv.style.display = "none";
        userInfoDiv.innerHTML = "";
      }
      if (mainContentDiv) mainContentDiv.style.display = "none";
      if (authRequiredMessageDiv)
        authRequiredMessageDiv.style.display = "block";
    }
  }

  // Check initial login state from storage on load
  chrome.storage.local.get(["isLoggedIn", "user"], (result) => {
    console.log("Popup: Initial auth state from storage:", result);
    updateUI(result.isLoggedIn, result.user);
  });

  // Listener for Sign In button
  if (signinButton) {
    signinButton.addEventListener("click", function () {
      console.log("Popup: Sign in button clicked");
      const extensionId = chrome.runtime.id;
      const callbackUrl = encodeURIComponent(
        `http://localhost:3000/auth/extension-callback`
      );
      // Use provider-specific signin if needed (e.g., Google)
      // const signInUrl = `http://localhost:3000/api/auth/signin/google?callbackUrl=${callbackUrl}`;
      // Or use the default signin page
      // Open signin page in new tab
      chrome.tabs.create({
        url: `http://localhost:3000/api/auth/signin?callbackUrl=${callbackUrl}`,
        active: true,
      });
    });
  }

  // Add message listener for the extension callback
  window.addEventListener("message", function (event) {
    console.log("Message received:", event);

    // In development, we may need to be more permissive with origins
    // For production, you should restrict this to trusted origins
    if (event.origin !== "http://localhost:3000") {
      console.log("Message from unexpected origin:", event.origin);
      // Continue processing anyway during development
      // return; // Uncomment for production
    }

    if (event.data && event.data.type === "EXTENSION_AUTH_SUCCESS") {
      console.log("Auth success message received:", event.data);
      const session = event.data.session;

      if (!session || !session.user) {
        console.error("Invalid session data received:", session);
        return;
      }

      chrome.storage.local.set(
        { isLoggedIn: true, user: session.user },
        function () {
          console.log("User data saved to storage:", session.user);
          updateUI(true, session.user);

          // Close the auth tab
          chrome.tabs.query(
            { url: "*://localhost:3000/auth/extension-callback*" },
            function (tabs) {
              if (tabs && tabs.length > 0) {
                chrome.tabs.remove(tabs[0].id).catch((error) => {
                  console.log("Error closing tab:", error.message);
                  // Tab might be already closed, this is fine
                });
              } else {
                console.warn("No callback tab found to close");
              }
            }
          );
        }
      );
    }
  });

  // Listener for Sign Out button
  if (signoutButton) {
    signoutButton.addEventListener("click", function () {
      console.log("Popup: Sign out button clicked");
      chrome.storage.local.remove(["isLoggedIn", "user"], () => {
        console.log("Popup: Cleared local auth state.");
        updateUI(false, null);
        // Optional: Clear server session cookie
        chrome.tabs.create({
          url: "http://localhost:3000/api/auth/signout",
          active: false,
        });
      });
    });
  }

  // Listener for messages from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("Popup: Message received:", message);
    if (message.action === "loginStatusChanged") {
      console.log(
        "Popup: loginStatusChanged message received. Re-checking storage..."
      );
      // Re-check storage when notified by background script
      chrome.storage.local.get(["isLoggedIn", "user"], (result) => {
        console.log(
          "Popup: Auth state updated from background message:",
          result
        );
        updateUI(result.isLoggedIn, result.user);
      });
      // It's good practice to send a response if the message handler might be async in the future
      // sendResponse({ received: true });
    }
    // Return true if you intend to use sendResponse asynchronously, otherwise it's optional
    // return true;
  });

  // --- Generate Cover Letter Logic ---
  // Check if we have a stored job description (moved inside DOMContentLoaded)
  chrome.storage.local.get(
    ["jobDescription", "currentJobPage"],
    function (data) {
      if (data.jobDescription) {
        const jobDescElement = document.getElementById("job-description");
        if (jobDescElement) {
          jobDescElement.value = data.jobDescription;
        }
        // Clear it so it's not used again unless updated
        chrome.storage.local.remove("jobDescription");
      }

      // Add event listener to the "Generate Cover Letter" button
      const generateButton = document.querySelector("#generate-btn"); // Use ID selector
      if (generateButton) {
        generateButton.addEventListener("click", function () {
          console.log("Popup: Generate button clicked");
          const jobDesc =
            document.getElementById("job-description")?.value || "";
          const highlights =
            document.getElementById("resume-highlights")?.value || "";
          console.log("Popup: Job Description:", jobDesc);
          console.log("Popup: Highlights:", highlights);

          chrome.storage.local.get(["isLoggedIn"], (result) => {
            if (result.isLoggedIn) {
              // Proceed with generation - TODO: Implement actual generation call
              alert("Generating cover letter... (Implement actual logic)");
              // Example: Send to background script
              // chrome.runtime.sendMessage({
              //   action: 'generateCoverLetter',
              //   jobDesc: jobDesc,
              //   highlights: highlights
              // }, response => { /* handle response */ });
            } else {
              alert("Please sign in first.");
            }
          });
        });
      } else {
        console.warn("Popup: Generate button not found."); // Add warning if button missing
      }
    }
  );
  // --- End Generate Cover Letter Logic ---
});
