// Linguistic Passages App
// Minimal JavaScript for superscript conversion and enhancements

// Convert ^{...} to <sup>...</sup> in all elements with .ipa-convert class
function convertSuperscripts() {
    document.querySelectorAll('.ipa-convert').forEach(el => {
        if (el.innerHTML.includes('^{')) {
            el.innerHTML = el.innerHTML.replace(/\^\{([^}]*)\}/g, '<sup>$1</sup>');
        }
    });
}

// Run on page load
document.addEventListener('DOMContentLoaded', convertSuperscripts);
