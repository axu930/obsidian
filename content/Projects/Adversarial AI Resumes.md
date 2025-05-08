A fun project to game AI resume screeners -- mostly GPT style screeners, but also keyword matchers.  The baseline plan is to generate resumes with 0.1-pt font white text that has mildly embellished qualifications, and see what changes make AI generate disproportionately good responses. 

# Current model
- Based off of [RenderCV](https://github.com/rendercv/rendercv)
- Modified the template files to insert 0.1 point white text after individual sections

# Ideas
- Put in 'keywords' and 'summaries' after resume sections in order to provide a favorable AI overview
- Encode AI prompt overrides with hexadecimal encoding so humans do not catch them
- 