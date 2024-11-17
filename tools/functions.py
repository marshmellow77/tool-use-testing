from tools.function_registry import Function, FunctionParameter, FunctionRegistry

# Initialize the registry
registry = FunctionRegistry()

# Register all functions
registry.register(Function(
    name="get_weather",
    description="Get the weather in a given location",
    parameters=[
        FunctionParameter(
            name="location",
            type="string",
            description="The city name of the location for which to get the weather."
        )
    ]
))

registry.register(Function(
    name="get_current_time",
    description="Get the current time in a specified timezone.",
    parameters=[
        FunctionParameter(
            name="timezone",
            type="string",
            description="The timezone to get the current time for, e.g., 'America/New_York'."
        )
    ]
))

registry.register(Function(
    name="translate_text",
    description="Translate a given text to a target language.",
    parameters=[
        FunctionParameter(
            name="text",
            type="string",
            description="The text to translate."
        ),
        FunctionParameter(
            name="target_language",
            type="string",
            description="The language to translate the text into, e.g., 'Spanish'."
        )
    ]
))

registry.register(Function(
    name="calculate_route",
    description="Calculate the best route between two locations.",
    parameters=[
        FunctionParameter(
            name="start_location",
            type="string",
            description="The starting point of the route."
        ),
        FunctionParameter(
            name="end_location",
            type="string",
            description="The destination point of the route."
        ),
        FunctionParameter(
            name="transport_mode",
            type="string",
            description="Mode of transport, e.g., 'driving', 'walking', 'bicycling'.",
            required=False
        )
    ]
))

registry.register(Function(
    name="book_hotel",
    description="Book a hotel room in a specified city.",
    parameters=[
        FunctionParameter(
            name="city",
            type="string",
            description="The city where the hotel is located."
        ),
        FunctionParameter(
            name="check_in_date",
            type="string",
            description="Check-in date in 'YYYY-MM-DD' format."
        ),
        FunctionParameter(
            name="check_out_date",
            type="string",
            description="Check-out date in 'YYYY-MM-DD' format."
        ),
        FunctionParameter(
            name="guests",
            type="integer",
            description="Number of guests staying.",
            required=False
        )
    ]
))

registry.register(Function(
    name="get_stock_price",
    description="Retrieve the current stock price for a given company symbol.",
    parameters=[
        FunctionParameter(
            name="symbol",
            type="string",
            description="Stock ticker symbol, e.g., 'AAPL' for Apple Inc."
        )
    ]
))

registry.register(Function(
    name="send_email",
    description="Send an email to a specified recipient.",
    parameters=[
        FunctionParameter(
            name="recipient",
            type="string",
            description="Email address of the recipient."
        ),
        FunctionParameter(
            name="subject",
            type="string",
            description="Subject line of the email."
        ),
        FunctionParameter(
            name="message",
            type="string",
            description="Body content of the email."
        )
    ]
))

registry.register(Function(
    name="get_news_headlines",
    description="Fetch the latest news headlines for a specific topic.",
    parameters=[
        FunctionParameter(
            name="topic",
            type="string",
            description="The news topic to search for, e.g., 'technology', 'sports'."
        ),
        FunctionParameter(
            name="language",
            type="string",
            description="Language of the news articles, e.g., 'en' for English.",
            required=False
        )
    ]
))

registry.register(Function(
    name="convert_units",
    description="Convert a value from one unit to another.",
    parameters=[
        FunctionParameter(
            name="value",
            type="number",
            description="The numerical value to convert."
        ),
        FunctionParameter(
            name="from_unit",
            type="string",
            description="The unit to convert from, e.g., 'meters'."
        ),
        FunctionParameter(
            name="to_unit",
            type="string",
            description="The unit to convert to, e.g., 'feet'."
        )
    ]
))

registry.register(Function(
    name="set_reminder",
    description="Set a reminder for a specific event at a given time.",
    parameters=[
        FunctionParameter(
            name="event",
            type="string",
            description="Description of the event to be reminded about."
        ),
        FunctionParameter(
            name="datetime",
            type="string",
            description="Date and time for the reminder in ISO 8601 format."
        )
    ]
))

registry.register(Function(
    name="find_restaurant",
    description="Find restaurants based on cuisine and location.",
    parameters=[
        FunctionParameter(
            name="cuisine_type",
            type="string",
            description="Type of cuisine desired, e.g., 'Italian', 'Mexican'."
        ),
        FunctionParameter(
            name="location",
            type="string",
            description="Geographical location to search in."
        ),
        FunctionParameter(
            name="price_range",
            type="string",
            description="Desired price range, e.g., '$', '$$', '$$$'.",
            required=False
        )
    ]
))

registry.register(Function(
    name="get_calendar_events",
    description="Get events from the calendar for a specific date.",
    parameters=[
        FunctionParameter(
            name="date",
            type="string",
            description="The date to retrieve events for, in 'YYYY-MM-DD' format."
        )
    ]
))

registry.register(Function(
    name="solve_math_problem",
    description="Solve a given math problem.",
    parameters=[
        FunctionParameter(
            name="problem",
            type="string",
            description="The math problem to solve."
        )
    ]
))

registry.register(Function(
    name="lookup_definition",
    description="Look up the definition of a word.",
    parameters=[
        FunctionParameter(
            name="word",
            type="string",
            description="The word to define."
        )
    ]
))

registry.register(Function(
    name="check_traffic",
    description="Check traffic conditions for a specific route.",
    parameters=[
        FunctionParameter(
            name="route",
            type="string",
            description="Description of the route to check."
        )
    ]
))

# Export the registry
ALL_FUNCTIONS = registry