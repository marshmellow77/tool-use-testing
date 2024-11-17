from vertexai.generative_models import FunctionDeclaration

ALL_FUNCTIONS = [
    FunctionDeclaration(
        name="get_weather",
        description="Get the weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name of the location for which to get the weather."}
            },
            "required": ["location"],
        },
    ),
    
    FunctionDeclaration(
        name="get_current_time",
        description="Get the current time in a specified timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get the current time for, e.g., 'America/New_York'."
                }
            },
            "required": ["timezone"],
        },
    ),
    
    FunctionDeclaration(
        name="translate_text",
        description="Translate a given text to a target language.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate."},
                "target_language": {"type": "string", "description": "The language to translate the text into, e.g., 'Spanish'."}
            },
            "required": ["text", "target_language"],
        },
    ),
    
    FunctionDeclaration(
        name="calculate_route",
        description="Calculate the best route between two locations.",
        parameters={
            "type": "object",
            "properties": {
                "start_location": {"type": "string", "description": "The starting point of the route."},
                "end_location": {"type": "string", "description": "The destination point of the route."},
                "transport_mode": {"type": "string", "description": "Mode of transport, e.g., 'driving', 'walking', 'bicycling'."}
            },
            "required": ["start_location", "end_location"],
        },
    ),
    
    FunctionDeclaration(
        name="book_hotel",
        description="Book a hotel room in a specified city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city where the hotel is located."},
                "check_in_date": {"type": "string", "description": "Check-in date in 'YYYY-MM-DD' format."},
                "check_out_date": {"type": "string", "description": "Check-out date in 'YYYY-MM-DD' format."},
                "guests": {"type": "integer", "description": "Number of guests staying."}
            },
            "required": ["city", "check_in_date", "check_out_date"],
        },
    ),
    
    FunctionDeclaration(
        name="get_stock_price",
        description="Retrieve the current stock price for a given company symbol.",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g., 'AAPL' for Apple Inc."}
            },
            "required": ["symbol"],
        },
    ),
    
    FunctionDeclaration(
        name="send_email",
        description="Send an email to a specified recipient.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Email address of the recipient."},
                "subject": {"type": "string", "description": "Subject line of the email."},
                "message": {"type": "string", "description": "Body content of the email."}
            },
            "required": ["recipient", "subject", "message"],
        },
    ),
    
    FunctionDeclaration(
        name="get_news_headlines",
        description="Fetch the latest news headlines for a specific topic.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The news topic to search for, e.g., 'technology', 'sports'."},
                "language": {"type": "string", "description": "Language of the news articles, e.g., 'en' for English."}
            },
            "required": ["topic"],
        },
    ),
    
    FunctionDeclaration(
        name="convert_units",
        description="Convert a value from one unit to another.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numerical value to convert."},
                "from_unit": {"type": "string", "description": "The unit to convert from, e.g., 'meters'."},
                "to_unit": {"type": "string", "description": "The unit to convert to, e.g., 'feet'."}
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    ),
    
    FunctionDeclaration(
        name="set_reminder",
        description="Set a reminder for a specific event at a given time.",
        parameters={
            "type": "object",
            "properties": {
                "event": {"type": "string", "description": "Description of the event to be reminded about."},
                "datetime": {"type": "string", "description": "Date and time for the reminder in ISO 8601 format."}
            },
            "required": ["event", "datetime"],
        },
    ),
    
    FunctionDeclaration(
        name="find_restaurant",
        description="Find restaurants based on cuisine and location.",
        parameters={
            "type": "object",
            "properties": {
                "cuisine_type": {"type": "string", "description": "Type of cuisine desired, e.g., 'Italian', 'Mexican'."},
                "location": {"type": "string", "description": "Geographical location to search in."},
                "price_range": {"type": "string", "description": "Desired price range, e.g., '$', '$$', '$$$'."}
            },
            "required": ["cuisine_type", "location"],
        },
    ),
    
    FunctionDeclaration(
        name="get_calendar_events",
        description="Get events from the calendar for a specific date.",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The date to retrieve events for, in 'YYYY-MM-DD' format."}
            },
            "required": ["date"],
        },
    ),
    
    FunctionDeclaration(
        name="solve_math_problem",
        description="Solve a given math problem.",
        parameters={
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The math problem to solve."}
            },
            "required": ["problem"],
        },
    ),
    
    FunctionDeclaration(
        name="lookup_definition",
        description="Look up the definition of a word.",
        parameters={
            "type": "object",
            "properties": {
                "word": {"type": "string", "description": "The word to define."}
            },
            "required": ["word"],
        },
    ),
    
    FunctionDeclaration(
        name="check_traffic",
        description="Check traffic conditions for a specific route.",
        parameters={
            "type": "object",
            "properties": {
                "route": {"type": "string", "description": "Description of the route to check."}
            },
            "required": ["route"],
        },
    )
] 