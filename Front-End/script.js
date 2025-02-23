document.addEventListener("DOMContentLoaded", () => {
    const toggleForm = document.getElementById("toggleForm")
    const loginForm = document.getElementById("loginForm")
    const signupForm = document.getElementById("signupForm")
    const formTitle = document.getElementById("formTitle")
    const formDescription = document.getElementById("formDescription")
  
    toggleForm.addEventListener("change", function () {
      if (this.checked) {
        loginForm.style.display = "none"
        signupForm.style.display = "block"
        formTitle.textContent = "Create Account"
        formDescription.textContent = "Join the AI Construction platform"
      } else {
        loginForm.style.display = "block"
        signupForm.style.display = "none"
        formTitle.textContent = "Welcome Back!"
        formDescription.textContent = "Access your AI Construction dashboard"
      }
    })
  
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault()
      // Here you would typically send the login data to your server
      console.log("Login form submitted")
    })
  
    signupForm.addEventListener("submit", (e) => {
      e.preventDefault()
      // Here you would typically send the signup data to your server
      console.log("Signup form submitted")
    })
  })
  
  