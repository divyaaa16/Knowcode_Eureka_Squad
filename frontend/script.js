// Handle file upload
document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        // Here you would typically handle the file upload
        // For now, we'll just log it
        console.log('File selected:', file.name);
        
        // You can add your image processing logic here
        const reader = new FileReader();
        reader.onload = function(e) {
            // Handle the image data
            console.log('Image loaded');
        };
        reader.readAsDataURL(file);
    }
});

// Responsive design
window.addEventListener('resize', function() {
    const width = window.innerWidth / 2;
    const height = window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Wrap all modal functionality in DOMContentLoaded event
document.addEventListener('DOMContentLoaded', () => {
    const signInModal = document.getElementById('signInModal');
    const signUpModal = document.getElementById('signUpModal');
    const signInBtn = document.querySelector('.sign-in-btn');
    const signUpBtn = document.querySelector('.sign-up-btn');
    const closeBtns = document.querySelectorAll('.close');
    const showSignUpLink = document.getElementById('showSignUp');
    const showSignInLink = document.getElementById('showSignIn');
    const nav = document.querySelector('nav');

    function openModal(modal) {
        modal.style.display = 'flex';
        setTimeout(() => {
            modal.classList.add('active');
        }, 10);
    }

    function closeModal(modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.style.display = 'none';
        }, 300);
    }

    function closeAllModals() {
        [signInModal, signUpModal].forEach(modal => closeModal(modal));
    }

    // Open modals with animation
    signInBtn.addEventListener('click', () => openModal(signInModal));
    signUpBtn.addEventListener('click', () => openModal(signUpModal));

    // Close modals with animation
    closeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            closeAllModals();
        });
    });

    // Switch between modals with animation
    showSignUpLink.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal(signInModal);
        setTimeout(() => {
            openModal(signUpModal);
        }, 300);
    });

    showSignInLink.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal(signUpModal);
        setTimeout(() => {
            openModal(signInModal);
        }, 300);
    });

    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === signInModal || e.target === signUpModal) {
            closeAllModals();
        }
    });

    // Form submission with loading animation
    const forms = document.querySelectorAll('#signInForm, #signUpForm');
    forms.forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = form.querySelector('.submit-btn');
            submitBtn.classList.add('loading');

            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            submitBtn.classList.remove('loading');
            console.log(`${form.id} submitted`);
            closeAllModals();
        });
    });

    // Add hover effect to social buttons
    const socialBtns = document.querySelectorAll('.social-buttons button');
    socialBtns.forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'translateY(-2px)';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translateY(0)';
        });
    });

    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    const chatbotIcon = document.getElementById('chatbotIcon');
    const chatbotContainer = document.getElementById('chatbotContainer');
    const closeChatbot = document.getElementById('closeChatbot');
    const userInput = document.getElementById('userInput');

    chatbotIcon.addEventListener('click', () => {
        chatbotContainer.classList.add('active');
    });

    closeChatbot.addEventListener('click', () => {
        chatbotContainer.classList.remove('active');
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

async function sendMessage() {
    const userInput = document.getElementById("userInput");
    const message = userInput.value.trim();
    if (!message) return;

    const chatBox = document.getElementById("chatBox");
    
    // Add user message
    chatBox.innerHTML += `<p class="user-message">${message}</p>`;
    userInput.value = '';

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                model: "gpt-4o-mini",
                store: true,
                messages: [
                    { role: "user", content: message }
                ]
            })
        });
        
        const data = await response.json();

        // Add bot response
        if (data.choices && data.choices.length > 0) {
            chatBox.innerHTML += `<p class="bot-message">${data.choices[0].message.content}</p>`;
        } else {
            chatBox.innerHTML += `<p class="bot-message">I apologize, but I couldn't process your request at the moment.</p>`;
        }
    } catch (error) {
        chatBox.innerHTML += `<p class="bot-message">I apologize, but there seems to be a technical issue. Please try again later.</p>`;
        console.error("Error:", error);
    }

    // Scroll to bottom again after bot responds
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Add your API key as a constant
const API_KEY = "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";
// Social login handling
document.querySelectorAll('.google-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Add your Google sign in logic here
        console.log('Google sign in clicked');
    });
});

document.querySelectorAll('.facebook-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Add your Facebook sign in logic here
        console.log('Facebook sign in clicked');
    });
}); 