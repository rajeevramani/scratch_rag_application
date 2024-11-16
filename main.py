
import asyncio
from scratch_rag_application.utils.logging_config import setup_logging
import logging
from scratch_rag_application.loader.url_loader import URLLoader


setup_logging()


async def main():
    logger = logging.getLogger(__name__)
    url_loader = URLLoader()
    docs = await url_loader.load_urls()
    logger.info(f"docs: {docs[:1]}")


if __name__ == "__main__":
    asyncio.run(main())
    # Get logger for this file
    # Step 1: Request the page content
    # url = "https://docs.konghq.com/konnect/api-products/"
    # response = requests.get(url)
    # soup = BeautifulSoup(response.text, "html.parser")
    #
    # # Extracting the main content container
    # main_content_div = soup.find("div", class_="page-content")
    #
    # logger.info(f"main content: {md(str(main_content_div))}")
    # for text in text_content:
    #     print(text)

    # # Sample YAML content
    # sample_yaml = """
    # database:
    #   credentials:
    #     username: admin
    #     password: secret123
    #   settings:
    #     host: localhost
    #     port: 5432
    #     pools:
    #       - name: main
    #         size: 10
    #       - name: backup
    #         size: 5
    #
    # application:
    #   name: MyApp
    #   version: 1.0.0
    #   features:
    #     - logging
    #     - authentication
    #     - monitoring
    # """
    #
    # # Create a temporary YAML file
    # with open('config.yaml', 'w') as f:
    #     f.write(sample_yaml)
    #
    # # Initialize handler
    # yaml_handler = ConfigHandler('config.yaml')
    #
    # # Examples of using the class
    # print("Database username:", yaml_handler.get(
    #     'database.credentials.username'))
    # print("First pool name:", yaml_handler.get(
    #     'database.settings.pools.0.name'))
    #
    # # Set a new value
    # yaml_handler.set('application.version', '1.1.0')
    #
    # # Get all possible paths
    # print("\nAll paths in YAML:")
    # for path in yaml_handler.get_all_paths():
    #     print(f"- {path}: {yaml_handler.get(path)}")
    #
    # # Save modifications
    # yaml_handler.save('config_modified.yaml')
