from typing import Any, Dict, List, Optional, Tuple
import re

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

icl_examples = {
    "flai": """Example 1:
User query: give me info about new music
Plan step 1:  get music using with sorting new 
API response: Successfully called GET /api/datapoint/sounds?sorting=new&take=20
{{"musicId": "7261315622624463621", "title": "sonido original", "creator": "Moto Fellowship"},
{"musicId": "7285470051778497282", "title": "оригинальный звук", "creator": "Саня Козлов",}}
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: List of new music: 1. Sound name is "sonido original" ("title") created by "Moto Fellowship" ("creator") with id 7261315622624463621 ("musicId")
                                 2. Sound name is "оригинальный звук" ("title") created by "Саня Козлов" ("creator") with id 7285470051778497282 ("musicId")

Example 2:
User query: Give me information about Khaby Lame?
Plan step 1: Search for the most popular authors using parameter Search="khaby.lame"
API response: Successfully called GET /api/datapoint/authors?take=1&Search=khaby.lame&Sorting=popularity
{"authorId": "6917544880626254853", "subscribers": 171400, "nickname": "Khaby.lame"}
Plan step 2: Search for more detail information about user with authorId=6917544880626254853
API response: Successfully called GET api/datapoint/author?authorId=6917544880626254853 
{"authorId": "6917544880626254853", "subscribers": 161500000, "likes": 2400000000, "nickname": "Khabane lame"}
Plan step 3: Analize full information about author, return information and your summorize  
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Khabane Lame has wery good account with 161500000 subscribers.

Example 3:
User query: Can you tell me url the author who has the most reposted music?
Plan step 1: get sounds using with sorting reposts and take 1, beacuse need only one most reposts author.
API response: Successfully called GET api/datapoint/sounds?sorting=reposts&take=1
{"musicId": "7098939394085702406", "reposts": 35200000,  "creator": "WZ Beat", "title": "Beat Automotivo Tan Tan Tan Viral"}
Plan step 2: search for the author by parameter Search="WZ Beat"
API response: Successfully called GET /api/datapoint/authors?take=10&Search=WZ%20Beat
{"authorId": "6664511694575157253", "likes": 1200000, "nickname": "WZ Beat"}
Plan step 3: request the detail information about author with authorId = "6664511694575157253"
API response: Successfully called GET /api/datapoint/author?authorId=6664511694575157253
{"authorId": "6664511694575157253", "subscribers": 239100, "url": "https://www.tiktok.com/@6664511694575157253", "nickname": "WZ Beat"}
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: The most reposted sound  is "Beat Automotivo Tan Tan Tan Viral" by WZ Beat, here is his url: https://www.tiktok.com/@6664511694575157253
""",
    "tmdb": """Example 1:
User query: give me some movies performed by Tony Leung.
Plan step 1: search person with name "Tony Leung"
API response: Tony Leung's person_id is 1337
Plan step 2: collect the list of movies performed by Tony Leung whose person_id is 1337
API response: Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Tony Leung has performed in Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero

Example 2:
User query: Who wrote the screenplay for the most famous movie directed by Martin Scorsese?
Plan step 1: search for the most popular movie directed by Martin Scorsese
API response: Successfully called GET /search/person to search for the director "Martin Scorsese". The id of Martin Scorsese is 1032
Plan step 2: Continue. search for the most popular movie directed by Martin Scorsese (1032)
API response: Successfully called GET /person/{{person_id}}/movie_credits to get the most popular movie directed by Martin Scorsese. The most popular movie directed by Martin Scorsese is Shutter Island (11324)
Plan step 3: search for the screenwriter of Shutter Island
API response: The screenwriter of Shutter Island is Laeta Kalogridis (20294)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Laeta Kalogridis wrote the screenplay for the most famous movie directed by Martin Scorsese.
""",
    "spotify": """Example 1:
User query: set the volume to 20 and skip to the next track.
Plan step 1: set the volume to 20
API response: Successfully called PUT /me/player/volume to set the volume to 20.
Plan step 2: skip to the next track
API response: Successfully called POST /me/player/next to skip to the next track.
Thought: I am finished executing a plan and completed the user's instructions
Final Answer: I have set the volume to 20 and skipped to the next track.

Example 2:
User query: Make a new playlist called "Love Coldplay" containing the most popular songs by Coldplay
Plan step 1: search for the most popular songs by Coldplay
API response: Successfully called GET /search to search for the artist Coldplay. The id of Coldplay is 4gzpq5DPGxSnKTe4SA8HAU
Plan step 2: Continue. search for the most popular songs by Coldplay (4gzpq5DPGxSnKTe4SA8HAU)
API response: Successfully called GET /artists/4gzpq5DPGxSnKTe4SA8HAU/top-tracks to get the most popular songs by Coldplay. The most popular songs by Coldplay are Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b).
Plan step 3: make a playlist called "Love Coldplay"
API response: Successfully called GET /me to get the user id. The user id is xxxxxxxxx.
Plan step 4: Continue. make a playlist called "Love Coldplay"
API response: Successfully called POST /users/xxxxxxxxx/playlists to make a playlist called "Love Coldplay". The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Plan step 5: Add the most popular songs by Coldplay, Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b), to playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T)
API response: Successfully called POST /playlists/7LjHVU3t3fcxj5aiPFEW4T/tracks to add Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b) in playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T). The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have made a new playlist called "Love Coldplay" containing Yellow and Viva La Vida by Coldplay.
"""
}


PLANNER_PROMPT = """You are an agent that plans solution to user queries.
You should always give your plan in natural language.
Another model will receive your plan and find the right API calls and give you the result in natural language.
If you assess that the current plan has not been fulfilled, you can output "Continue" to let the API selector select another API to fulfill the plan.
If you think you have got the final answer or the user query has been fulfilled, just output the answer immediately. If the query has not been fulfilled, you should continue to output your plan.
In most case, search, filter, and sort should be completed in a single step.
The plan should be as specific as possible. It is better not to use pronouns in plan, but to use the corresponding results obtained previously. For example, instead of "Get the most popular movie directed by this person", you should output "Get the most popular movie directed by Martin Scorsese (1032)". If you want to iteratively query something about items in a list, then the list and the elements in the list should also appear in your plan.
The plan should be straightforward. If you want to search, sort or filter, you can put the condition in your plan. For example, if the query is "Who is the lead actor of In the Mood for Love (id 843)", instead of "get the list of actors of In the Mood for Love", you should output "get the lead actor of In the Mood for Love (843)".

Starting below, you should follow this format:

User query: the query a User wants help with related to the API.
Plan step 1: the first step of your plan for how to solve the query
API response: the result of executing the first step of your plan, including the specific API call made.
Plan step 2: based on the API response, the second step of your plan for how to solve the query. If the last step result is not what you want, you can output "Continue" to let the API selector select another API to fulfill the plan. For example, the last plan is "add a song (id xxx) in my playlist", but the last step API response is calling "GET /me/playlists" and getting the id of my playlist, then you should output "Continue" to let the API selector select another API to add the song to my playlist. Pay attention to the specific API called in the last step API response. If a inproper API is called, then the response may be wrong and you should give a new plan.
API response: the result of executing the second step of your plan
... (this Plan step n and API response can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: the final output from executing the plan


{icl_examples}

Begin!

User query: {input}
Plan step 1: {agent_scratchpad}"""




class Planner(Chain):
    llm: BaseLLM
    scenario: str
    planner_prompt: str
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, scenario: str, planner_prompt=PLANNER_PROMPT) -> None:
        super().__init__(llm=llm, scenario=scenario, planner_prompt=planner_prompt)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Planner"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "API response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Plan step {}: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]
    
    def _construct_scratchpad(
        self, history: List[Tuple[str, str]]
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += self.llm_prefix.format(i + 1) + plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        return scratchpad

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        scratchpad = self._construct_scratchpad(inputs['history'])
        # print("Scrachpad: \n", scratchpad)
        planner_prompt = PromptTemplate(
            template=self.planner_prompt,
            partial_variables={
                "agent_scratchpad": scratchpad,
                "icl_examples": icl_examples[self.scenario],
            },
            input_variables=["input"]
        )
        planner_chain = LLMChain(llm=self.llm, prompt=planner_prompt)
        planner_chain_output = planner_chain.run(input=inputs['input'], stop=self._stop)

        planner_chain_output = re.sub(r"Plan step \d+: ", "", planner_chain_output).strip()

        return {"result": planner_chain_output}
