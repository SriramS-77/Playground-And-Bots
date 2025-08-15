import asyncio
import random
import math
import time
from playwright.async_api import async_playwright

# --- Configuration ---
PLAYGROUND_URL = "http://127.0.0.1:5000"

# DEFAULT_CONFIG = {
#     "scroll_amount": 150,  # Pixels per scroll tick. Smaller is slower.
#     "scroll_delay_min": 0.03,
#     "scroll_delay_max": 0.08,
#     "typing_delay_min": 0.04,
#     "typing_delay_max": 0.15,
# }

DEFAULT_CONFIG = {
    "scroll_amount": 30,  # Pixels per scroll tick. Smaller is slower.
    "scroll_delay_min": 0.19,
    "scroll_delay_max": 0.28,
    "typing_delay_min": 0.18,
    "typing_delay_max": 0.24,
}

# --- Behavior Modules (for advanced bots) ---
def generate_bezier_path(start_point, end_point, control_point_offset=50):
    """
    Generates a series of points along a quadratic Bézier curve.
    This creates a more natural, curved mouse path.
    """
    points = []
    # Introduce randomness in control points to vary the curve
    control_x = (start_point[0] + end_point[0]) / 2 + random.uniform(-control_point_offset, control_point_offset)
    control_y = (start_point[1] + end_point[1]) / 2 + random.uniform(-control_point_offset, control_point_offset)

    for t in range(0, 101, 5):  # 20 steps for a smooth path
        t_decimal = t / 100
        x = ((1 - t_decimal) ** 2 * start_point[0]) + (2 * (1 - t_decimal) * t_decimal * control_x) + (
                (t_decimal ** 2) * end_point[0])
        y = ((1 - t_decimal) ** 2 * start_point[1]) + (2 * (1 - t_decimal) * t_decimal * control_y) + (
                (t_decimal ** 2) * end_point[1])
        points.append((x, y))
    return points


# --- Base Bot Class (Refactored for Robustness) ---
class BaseBot:
    """
    Handles the browser lifecycle and navigation using robust context managers.
    """

    def __init__(self, bot_id, config=None):
        self.bot_id = bot_id
        self.page = None
        if config is None:
            config = {}
        # Merge provided config with defaults
        self.config = {**DEFAULT_CONFIG, **config}

    async def surf_website(self):
        """Defines the bot's browsing behavior. To be implemented by subclasses."""
        raise NotImplementedError

    async def run(self):
        """
        The main execution loop for the bot. It handles the entire browser
        lifecycle using async context managers for improved robustness and
        automatic cleanup.
        """
        async with async_playwright() as p:
            browser = None
            try:
                browser = await p.chromium.launch(headless=False)  # Run in headed mode to watch
                context = await browser.new_context(
                    viewport={'width': 1366, 'height': 768},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                self.page = await context.new_page()

                print(f"Bot {self.bot_id} ({self.__class__.__name__}): Starting session.")
                await self.page.goto(PLAYGROUND_URL)
                await self.page.wait_for_load_state('networkidle')

                # --- Inject Bot Metadata into the Page ---
                bot_metadata = {
                    'bot_id': self.bot_id,
                    'bot_name': self.__class__.__name__,
                    'session_timestamp': int(time.time() * 1000)
                }
                await self.page.evaluate(f"""
                    () => {{
                        if (window.userData) {{
                            window.userData.bot_info = {bot_metadata};
                        }}
                    }}
                """)

                await self.surf_website()

            except Exception as e:
                if "Target page, context or browser has been closed" not in str(e):
                    print(f"Bot {self.bot_id} ({self.__class__.__name__}): An error occurred: {e}")
            finally:
                if browser:
                    await browser.close()
                print(f"Bot {self.bot_id} ({self.__class__.__name__}): Session ended.")


# --- Tier 1: Naive Bot ---
class NaiveBot(BaseBot):
    """A simple bot with a predictable path and no attempt to hide its nature."""

    async def surf_website(self):
        # A simple, fixed path
        await self.page.click('nav a[href="#blog"]')
        await self.page.wait_for_selector("#search-input")  # Wait for blog page to render
        await asyncio.sleep(1)

        # Click the first "Read More" link it finds
        await self.page.locator('a:has-text("Read More")').first.click()
        await self.page.wait_for_selector("a:has-text('Back to Blog List')")  # Wait for detail page
        await asyncio.sleep(1)

        await self.page.click('nav a[href="#contact"]')
        await self.page.wait_for_selector("#contact-form")  # Wait for contact page
        await asyncio.sleep(1)

        await self.page.fill('#name', 'Test Bot')
        await self.page.fill('#email', 'bot@test.com')
        await self.page.fill('#message', 'This is a test message.')
        await self.page.click('button[type="submit"]')
        await asyncio.sleep(2)

        await self.page.close()


# --- Tier 2: Humanish Bot ---
class HumanishBot(BaseBot):
    """A bot that adds random delays, simple mouse movements, and basic random exploration."""

    async def move_and_click(self, locator):
        await locator.scroll_into_view_if_needed()
        await locator.wait_for(state='visible')
        box = await locator.bounding_box()
        if box:
            target_x = box['x'] + box['width'] / 2
            target_y = box['y'] + box['height'] / 2
            await self.page.mouse.move(target_x, target_y, steps=5)
            await locator.click()  # Use the locator's robust click

    async def type_text(self, selector, text):
        min_delay = self.config["typing_delay_min"]
        max_delay = self.config["typing_delay_max"]
        for char in text:
            await self.page.type(selector, char)
            await asyncio.sleep(random.uniform(min_delay, max_delay))

    async def read_a_blog_post(self):
        """Simulates reading a blog post with scrolling and a chance to navigate."""
        print(f"Bot {self.bot_id}: Reading a blog post...")
        await asyncio.sleep(random.uniform(2, 4))
        # Scroll down the page
        for _ in range(random.randint(2, 5)):
            await self.page.mouse.wheel(0, random.randint(200, 500))
            await asyncio.sleep(random.uniform(0.5, 1.5))

        # Randomly decide to click next, previous, or go back
        action = random.choice(['next', 'previous', 'back', 'back'])
        if action == 'next':
            next_button = self.page.locator('a:has-text("Next")')
            if await next_button.is_visible():
                await self.move_and_click(next_button)
                await self.page.wait_for_selector("a:has-text('Back to Blog List')")
                await self.read_a_blog_post()  # Read the next post
        elif action == 'previous':
            prev_button = self.page.locator('a:has-text("Previous")')
            if await prev_button.is_visible():
                await self.move_and_click(prev_button)
                await self.page.wait_for_selector("a:has-text('Back to Blog List')")

    async def browse_blog_list(self):
        """Simulates browsing the main blog page and picking an article."""
        print(f"Bot {self.bot_id}: Browsing blog list...")
        await self.move_and_click(self.page.locator('nav a[href="#blog"]'))
        await self.page.wait_for_selector("#search-input")
        await asyncio.sleep(random.uniform(2, 4))

        # Scroll around a bit
        await self.page.mouse.wheel(0, random.randint(300, 600))
        await asyncio.sleep(random.uniform(1, 3))

        # Pick a random blog to read
        all_posts = await self.page.locator('a:has-text("Read More")').all()
        if all_posts:
            await self.move_and_click(random.choice(all_posts))
            await self.page.wait_for_selector("a:has-text('Back to Blog List')")
            await self.read_a_blog_post()

    async def fill_contact_form(self):
        print(f"Bot {self.bot_id}: Filling contact form...")
        await self.move_and_click(self.page.locator('nav a[href="#contact"]'))
        await self.page.wait_for_selector("#contact-form")
        await self.type_text('#name', 'Humanish Bot')
        await self.type_text('#email', 'humanish@test.com')
        await self.type_text('#message', 'This is a carefully typed test message.')
        await self.move_and_click(self.page.locator('button[type="submit"]'))
        await asyncio.sleep(random.uniform(2, 4))

    async def surf_website(self):
        """A random sequence of actions to simulate a session."""
        await asyncio.sleep(random.uniform(1, 3))

        # Perform a random number of high-level actions
        for _ in range(random.randint(1, 2)):
            action = random.choice([self.browse_blog_list, self.browse_blog_list, self.browse_blog_list, self.fill_contact_form])
            await action()
            # Go home between actions sometimes
            if random.random() < 0.5:
                await self.move_and_click(self.page.locator('nav a:has-text("Home")'))
                await self.page.wait_for_selector("h1:has-text('Welcome to the Playground')")

        await self.page.close()


# --- Tier 3 & 4 Common Logic ---
class AdvancedBot(BaseBot):
    """A base for advanced bots with complex movements and behaviors."""

    async def is_in_viewport(self, locator):
        """Checks if an element is currently within the browser's viewport."""
        box = await locator.bounding_box()
        if not box:
            return False
        viewport = self.page.viewport_size
        return (
                box['y'] >= 0 and
                box['y'] + box['height'] <= viewport['height']
        )

    async def scroll_to_element(self, locator):
        """A smooth, human-like scroll function."""
        scroll_amount = self.config["scroll_amount"]
        min_delay = self.config["scroll_delay_min"]
        max_delay = self.config["scroll_delay_max"]

        while not await self.is_in_viewport(locator):
            # print(self.page.viewport_size)
            box = await locator.bounding_box()
            if not box: return  # Element disappeared

            # Determine scroll direction
            if box['y'] < 0:  # Element is above the viewport
                await self.page.mouse.wheel(0, -scroll_amount)
            elif box['y'] + box['height'] > self.page.viewport_size['height']:  # Element is below
                await self.page.mouse.wheel(0, scroll_amount)

            await asyncio.sleep(random.uniform(min_delay, max_delay))

    async def move_and_click(self, locator):
        # print(locator)
        await self.scroll_to_element(locator)
        await locator.wait_for(state='visible')
        box = await locator.bounding_box()
        if box:
            current_mouse_pos = await self.page.evaluate(
                '() => window.mousePos || {x: window.innerWidth/2, y: window.innerHeight/2}')
            target_x = box['x'] + random.uniform(box['width'] * 0.2, box['width'] * 0.8)
            target_y = box['y'] + random.uniform(box['height'] * 0.2, box['height'] * 0.8)
            path = generate_bezier_path((current_mouse_pos['x'], current_mouse_pos['y']), (target_x, target_y))
            for x, y in path:
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.01)
            await locator.click()  # Use the locator's robust click

    async def type_text(self, selector, text):
        min_delay = self.config["typing_delay_min"]
        max_delay = self.config["typing_delay_max"]
        for char in text:
            await self.page.type(selector, char)
            if random.random() < 0.1:
                await asyncio.sleep(random.uniform(0.2, 0.5))
            else:
                await asyncio.sleep(random.uniform(min_delay, max_delay))

    async def read_a_blog_post(self):
        print(f"Bot {self.bot_id}: Reading a blog post...")
        await asyncio.sleep(random.uniform(3, 6))
        for _ in range(random.randint(4, 8)):
            await self.page.mouse.wheel(0, random.randint(100, 300))
            await asyncio.sleep(random.uniform(0.5, 1.5))

        for _ in range(random.randint(1, 3)):  # Chance to navigate multiple times
            action = random.choice(['next', 'previous', 'back'])
            if action == 'next':
                locator = self.page.locator('a:has-text("Next")')
                if await locator.is_visible():
                    await self.move_and_click(locator)
                    await self.page.wait_for_selector("a:has-text('Back to Blog List')")
                else:
                    break
            elif action == 'previous':
                locator = self.page.locator('a:has-text("Previous")')
                if await locator.is_visible():
                    await self.move_and_click(locator)
                    await self.page.wait_for_selector("a:has-text('Back to Blog List')")
                else:
                    break
            else:  # Go back to list
                break

    async def browse_blog_list(self):
        print(f"Bot {self.bot_id}: Browsing blog list...")
        await self.move_and_click(self.page.locator('nav a[href="#blog"]'))
        await self.page.wait_for_selector("#search-input")
        await asyncio.sleep(random.uniform(2, 4))

        if random.random() < 0.33:  # 33% chance to use search
            print(f"Bot {self.bot_id}: Using search...")
            search_term = random.choice(['security', 'network', 'data', 'web'])
            await self.type_text('#search-input', search_term)
            await asyncio.sleep(random.uniform(1, 3))

        all_posts = await self.page.locator('a:has-text("Read More")').all()
        visible_posts = [p for p in all_posts if await p.is_visible()]
        if visible_posts:
            await self.move_and_click(random.choice(visible_posts))
            await self.page.wait_for_selector("a:has-text('Back to Blog List')")
            await self.read_a_blog_post()

    async def fill_contact_form(self):
        print(f"Bot {self.bot_id}: Filling contact form...")
        await self.move_and_click(self.page.locator('nav a[href="#contact"]'))
        await self.page.wait_for_selector("#contact-form")
        await self.type_text('#name', 'Mimic Bot')
        await self.type_text('#email', 'mimic@test.com')
        await self.type_text('#message', 'This is a message to test the system.')
        await self.move_and_click(self.page.locator('button[type="submit"]'))
        await asyncio.sleep(random.uniform(2, 4))

    async def surf_website(self):
        await asyncio.sleep(random.uniform(2, 4))
        for _ in range(random.randint(2, 4)):
            action = random.choice(
                [self.browse_blog_list, self.fill_contact_form, self.browse_blog_list, self.browse_blog_list])  # Skew towards browsing
            await action()
            await self.move_and_click(self.page.locator('nav a:has-text("Home")'))
            await self.page.wait_for_selector("h1:has-text('Welcome to the Playground')")
        await self.page.close()


# --- Tier 3: Mimic Bot ---
class MimicBot(AdvancedBot):
    """An advanced bot that uses the complex, randomized exploration logic."""
    pass  # Inherits all the advanced logic


# --- Tier 4: Fallible Bot ---
class FallibleBot(AdvancedBot):
    """The most advanced bot, adding human-like errors to the advanced exploration logic."""
    KEYBOARD_MAP = {
        'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu', 'u': 'yhji',
        'i': 'ujko', 'o': 'iklp', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
        'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }

    async def scroll_to_element(self, locator, overshoot_chance=0.2):
        """A fallible, human-like scroll that can overshoot."""
        scroll_amount = self.config["scroll_amount"]
        min_delay = self.config["scroll_delay_min"]
        max_delay = self.config["scroll_delay_max"]
        needs_overshoot = random.random() < overshoot_chance
        overshot = False

        while not await self.is_in_viewport(locator):
            box = await locator.bounding_box()
            if not box: return

            scroll_direction = 0
            # print(box['y'], box['y'] + box['height'], self.page.viewport_size['height'])
            if box['y'] < 0:
                scroll_direction = -scroll_amount
            elif box['y'] + box['height'] > self.page.viewport_size['height']:
                scroll_direction = scroll_amount

            # Apply overshoot logic
            is_close_to_view = (scroll_direction > 0 and box['y'] < self.page.viewport_size['height'] * 1.5) or \
                               (scroll_direction < 0 and box['y'] + box['height'] > -self.page.viewport_size[
                                   'height'] * 0.5)

            if needs_overshoot and not overshot and is_close_to_view:
                print(f"Bot {self.bot_id}: Overshooting scroll...")
                overshoot_amount = scroll_direction * random.randint(2, 4)
                await self.page.mouse.wheel(0, overshoot_amount)
                overshot = True
                needs_overshoot = False  # Don't overshoot again
                await asyncio.sleep(random.uniform(0.2, 0.5))  # "Realization" pause
            else:
                await self.page.mouse.wheel(0, scroll_direction)

            await asyncio.sleep(random.uniform(min_delay, max_delay))

    async def type_text(self, selector, text, error_rate=0.15, correction_chance=0.8):
        """Overrides the base type_text to introduce errors."""
        min_delay = self.config["typing_delay_min"]
        max_delay = self.config["typing_delay_max"]
        flawed_text = ""
        errors = []
        i = 0
        while i < len(text):
            char = text[i]
            if char.lower() in self.KEYBOARD_MAP and random.random() < error_rate:
                error_type = random.choice(['substitution', 'insertion', 'deletion', 'transposition'])
                if error_type == 'substitution':
                    flawed_text += random.choice(self.KEYBOARD_MAP[char.lower()])
                    errors.append({'index': len(flawed_text) - 1, 'correct_char': char, 'type': 'substitution'})
                elif error_type == 'insertion':
                    flawed_text += random.choice(self.KEYBOARD_MAP[char.lower()])
                    flawed_text += char
                    errors.append({'index': len(flawed_text) - 2, 'type': 'insertion'})
                elif error_type == 'deletion':
                    errors.append({'index': len(flawed_text), 'correct_char': char, 'type': 'deletion'})
                elif error_type == 'transposition' and i + 1 < len(text):
                    flawed_text += text[i + 1] + text[i]
                    errors.append(
                        {'index': len(flawed_text) - 2, 'correct_pair': text[i:i + 2], 'type': 'transposition'})
                    i += 1
                else:
                    flawed_text += char
            else:
                flawed_text += char
            i += 1

        for char in flawed_text:
            await self.page.type(selector, char)
            await asyncio.sleep(random.uniform(min_delay, max_delay))

        await asyncio.sleep(random.uniform(1.0, 2.5))
        current_pos = len(flawed_text)
        for error in sorted(errors, key=lambda x: x['index'], reverse=True):
            if random.random() < correction_chance:
                distance = current_pos - error['index']
                for _ in range(distance):
                    await self.page.press(selector, 'ArrowLeft')
                    await asyncio.sleep(random.uniform(0.01, 0.03))

                if error['type'] == 'substitution':
                    await self.page.press(selector, 'Backspace')
                    await self.page.type(selector, error['correct_char'])
                    current_pos = error['index']
                elif error['type'] == 'insertion':
                    await self.page.press(selector, 'Backspace')
                    current_pos = error['index']
                elif error['type'] == 'deletion':
                    await self.page.type(selector, error['correct_char'])
                    current_pos = error['index'] + 1
                elif error['type'] == 'transposition':
                    await self.page.press(selector, 'Backspace')
                    await self.page.press(selector, 'Backspace')
                    await self.page.type(selector, error['correct_pair'])
                    current_pos = error['index']
                await asyncio.sleep(random.uniform(0.2, 0.6))
        await self.page.press(selector, 'End')

    async def move_and_click(self, locator, overshoot_chance=0.3):
        """Overrides the base move_and_click to add overshooting."""
        await self.scroll_to_element(locator)  # Use the fallible scroll
        await locator.wait_for(state='visible')
        box = await locator.bounding_box()
        if box:
            current_mouse_pos = await self.page.evaluate(
                '() => window.mousePos || {x: window.innerWidth/2, y: window.innerHeight/2}')
            target_x = box['x'] + random.uniform(box['width'] * 0.2, box['width'] * 0.8)
            target_y = box['y'] + random.uniform(box['height'] * 0.2, box['height'] * 0.8)
            final_target = (target_x, target_y)
            if random.random() < overshoot_chance:
                target_x += random.uniform(-30, 30)
                target_y += random.uniform(-30, 30)

            path = generate_bezier_path((current_mouse_pos['x'], current_mouse_pos['y']), (target_x, target_y))
            for x, y in path:
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.01)

            if (target_x, target_y) != final_target:
                await asyncio.sleep(random.uniform(0.1, 0.4))
                await self.page.mouse.move(final_target[0], final_target[1], steps=3)

            await locator.click()  # Use the locator's robust click


# --- Orchestrator ---
async def main():
    """
    Configure and run the bot simulations.
    """
    # Example of a slower, more deliberate configuration
    slower_config = {
        "scroll_amount": 25,  # Smaller scroll amount = slower scroll
        "scroll_delay_min": 0.22,
        "scroll_delay_max": 0.28,
        "typing_delay_min": 0.29,  # Higher min delay = slower typing
        "typing_delay_max": 0.35,
    }

    N = 3
    for _ in range(N):
        # await HumanishBot(bot_id=2).run()
        await MimicBot(bot_id=3).run()
        await FallibleBot(bot_id=4, config=slower_config).run()

        # tasks = [
        #     # NaiveBot(bot_id=1).run(),
        #     HumanishBot(bot_id=2).run(),
        #     MimicBot(bot_id=3).run(),
        #     FallibleBot(bot_id=4, config=slower_config).run(),
        # ]
        # await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
