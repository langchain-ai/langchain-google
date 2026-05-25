# Pull Request Instructions

## Changes Made
Added support for `{"type": "video", "url": "...", "mime_type": "video/mp4"}` format in `HumanMessage` content.

## Branch
`fix/video-url-support`

## Next Steps

### 1. Fork the repository (if you haven't already)
Go to https://github.com/langchain-ai/langchain-google and click "Fork"

### 2. Add your fork as a remote and push
```bash
cd /Users/apple/Documents/code/marky/marky-app/langchain-google

# Add your fork as a remote (replace YOUR_USERNAME with your GitHub username)
git remote add fork https://github.com/YOUR_USERNAME/langchain-google.git

# Push the branch to your fork
git push -u fork fix/video-url-support
```

### 3. Create the PR
After pushing, go to: https://github.com/langchain-ai/langchain-google/compare

Or use this direct link (after pushing):
https://github.com/langchain-ai/langchain-google/compare/main...YOUR_USERNAME:langchain-google:fix/video-url-support

## PR Title
```
fix(genai): Add support for video URL format in HumanMessage content
```

## PR Description
```markdown
## Problem
The documentation shows using `{"type": "video", "url": "...", "mime_type": "video/mp4"}` format in `HumanMessage` content, but the `_convert_to_parts` function was missing a handler for this format. This caused video URLs to be incorrectly processed by the `is_data_content_block` handler, which would download the YouTube page HTML as bytes instead of passing the URL as `file_uri` to Google's API.

## Solution
Added a handler for `"type": "video"` that:
1. Validates the `url` field is present
2. Extracts `mime_type` (defaults to "video/mp4")
3. Converts the URL to `FileData(file_uri=url, mime_type=mime_type)` format that Google's API expects

The handler is placed **before** the `is_data_content_block` check to ensure video dicts are handled correctly.

## Testing
Tested with `gemini-2.5-flash` model:
```python
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
message = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize the video in 3 sentences."},
        {
            "type": "video",
            "url": "https://www.youtube.com/watch?v=9hE5-98ZeCg",
            "mime_type": "video/mp4",
        },
    ]
)
response = model.invoke([message])
# ✅ Works correctly
```

## Related
Fixes the issue where the documented format from https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai didn't work.
```





